import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.models import CBraMod, CBraModClassifier


def synthetic_record(record, sampling_rate, **kwargs):
    if sampling_rate != 256:
        raise ValueError('Use 256 Hz source samples before independent patch resampling.')
    return {'eeg_data': np.random.RandomState(42).normal(size=(19, 4096)),
            'label': record[1], 'record': record}


class TestCBraModFinetuningExamples(unittest.TestCase):
    def example(self, kind):
        return Path(__file__).resolve().parents[2] / 'examples' / f'examples_eegmat_cbramod_{kind}.py'

    def test_full_training_keeps_mask_fixed_and_restores_complete_state(self):
        self.run_example('finetune')

    def test_bias_training_updates_bias_and_head_only(self):
        self.run_example('bitfit')

    def test_last_block_training_keeps_stem_and_early_blocks_fixed(self):
        self.run_example('last_blocks')

    def run_example(self, kind):
        captured = {}

        def observe(backbone, **kwargs):
            model = CBraModClassifier(backbone, **kwargs)
            captured['model'] = model
            captured['initial'] = {name: value.clone() for name, value in model.state_dict().items()}
            return model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            for subject in range(2):
                for condition in (1, 2):
                    (source / f'Subject{subject:02d}_{condition}.edf').write_bytes(b'synthetic source placeholder')
            output = root / 'model.pt'
            arguments = [str(self.example(kind)), '--root-path', str(source), '--cache-path', str(root / 'cache'),
                         '--epochs', '1', '--batch-size', '3', '--output-path', str(output)]
            if kind == 'last_blocks':
                arguments.extend(['--num-unfrozen-blocks', '1'])
            text = io.StringIO()
            with patch.object(MentalArithmeticDataset, 'read_record', side_effect=synthetic_record), \
                    patch('torcheeg.models.CBraModClassifier', side_effect=observe), \
                    patch('sys.argv', arguments), redirect_stdout(text):
                namespace = runpy.run_path(str(self.example(kind)), run_name='__main__')
            result = ast.literal_eval(text.getvalue().strip().splitlines()[-1])
            payload = torch.load(output, weights_only=True)
            self.assertEqual((result['train_windows'], result['test_windows']), (8, 8))
            self.assertEqual(payload['train_subjects'], ['Subject00'])
            self.assertEqual(payload['held_out_subject'], 'Subject01')
            self.assertFalse(payload['checkpoint_loaded'])
            self.assertFalse(payload['task_config']['freeze_encoder'])
            self.assertEqual(payload['sampling_rate'], 256)
            self.assertEqual(payload['target_patch_sampling_rate'], 200)
            model = captured['model']
            trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            self.assertEqual(payload['trainable_parameters'], trainable)
            self.assertFalse(model.backbone.patch_embedding.mask_encoding.requires_grad)
            changed = set()
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    torch.testing.assert_close(parameter, captured['initial'][name], rtol=0, atol=0)
                if not torch.equal(parameter, captured['initial'][name]):
                    changed.add(name)
            self.assertIn('task_head.output.weight', changed)
            self.assertIn('backbone.encoder.layers.1.self_attn_s.in_proj_bias', changed)
            if kind == 'finetune':
                self.assertIn('backbone.patch_embedding.proj_in.0.weight', changed)
                self.assertIn('backbone.encoder.layers.0.linear1.weight', changed)
            elif kind == 'bitfit':
                self.assertTrue(all(name.startswith('task_head.') or name.endswith('bias') for name in changed))
                self.assertTrue(all(name.startswith('task_head.') or name.endswith('bias') for name in trainable))
            else:
                self.assertTrue(all(name.startswith(('task_head.', 'backbone.encoder.layers.1.')) for name in trainable))
                self.assertIn('backbone.encoder.layers.1.linear1.weight', changed)
            # The retained native reconstruction head is not used for classification.
            for name in ('backbone.proj_out.0.weight', 'backbone.proj_out.0.bias'):
                torch.testing.assert_close(payload['state_dict'][name], captured['initial'][name], rtol=0, atol=0)
            restored = CBraModClassifier(CBraMod(**payload['backbone_config']), **payload['task_config'])
            if kind == 'bitfit':
                namespace['configure_trainable_parameters'](restored)
            elif kind == 'last_blocks':
                namespace['configure_trainable_parameters'](restored, payload['tuning_config']['num_unfrozen_blocks'])
            restored.load_state_dict(payload['state_dict'], strict=True)
            self.assertEqual([name for name, parameter in restored.named_parameters() if parameter.requires_grad], trainable)
            inputs = torch.linspace(-1, 1, 2 * 19 * 1024).reshape(2, 19, 1024)
            with torch.no_grad():
                torch.testing.assert_close(restored.eval()(inputs), model.eval()(inputs), rtol=0, atol=0)
            self.assertAlmostEqual(F.cross_entropy(payload['test_logits'], payload['test_targets']).item(),
                                   result['metrics']['test_loss'], places=5)
            restored.train()
            self.assertTrue(restored.backbone.training)
            self.assertTrue(all(block.training for block in restored.backbone.encoder.layers))

    def build(self):
        return CBraModClassifier(CBraMod(patch_size=25, hid_channels=16, mlp_channels=32,
            depth=3, num_heads=4, dropout=0., position_kernel=(3, 3)),
            num_electrodes=3, chunk_size=128, num_patches=4, num_classes=2, dropout=0.)

    def test_block_selection_validates_before_mutating_flags(self):
        configure = runpy.run_path(str(self.example('last_blocks')))['configure_trainable_parameters']
        model = self.build().eval()
        before = [(name, id(parameter), parameter.requires_grad) for name, parameter in model.named_parameters()]
        state = {name: value.clone() for name, value in model.state_dict().items()}
        for invalid in (-1, 0, True, 4, None, '1', 1.5):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'num_unfrozen_blocks'):
                configure(model, invalid)
            self.assertEqual(before, [(name, id(parameter), parameter.requires_grad) for name, parameter in model.named_parameters()])
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        for count in (1, 2, 3):
            configure(model, count)
            for index, block in enumerate(model.backbone.encoder.layers):
                self.assertEqual(all(p.requires_grad for p in block.parameters()), index >= 3 - count)
            self.assertFalse(any(p.requires_grad for p in model.backbone.patch_embedding.parameters()))
            self.assertFalse(model.training)

    def test_recipes_keep_native_keys_input_gradients_and_fixed_mask(self):
        for kind in ('bitfit', 'last_blocks'):
            with self.subTest(kind=kind):
                configure = runpy.run_path(str(self.example(kind)))['configure_trainable_parameters']
                model = self.build().double().eval()
                ids = {name: id(parameter) for name, parameter in model.named_parameters()}
                x = torch.randn(2, 3, 128, dtype=torch.float64, requires_grad=True)
                mask = torch.zeros(2, 3, 4, dtype=torch.bool)
                mask[:, :, 0] = True
                expected = model(x, mask=mask).detach()
                configure(model)
                self.assertEqual(ids, {name: id(parameter) for name, parameter in model.named_parameters()})
                actual = model(x, mask=mask)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.square().sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(x.grad[:, :, 32:].abs().sum().item(), 0.)
                torch.testing.assert_close(x.grad[:, :, :32], torch.zeros_like(x.grad[:, :, :32]), rtol=0, atol=0)
                self.assertFalse(model.backbone.patch_embedding.mask_encoding.requires_grad)
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        self.assertIsNone(parameter.grad, name)
                model.freeze_encoder = True
                flags = [p.requires_grad for p in model.parameters()]
                with self.assertRaisesRegex(ValueError, 'freeze_encoder=False'):
                    configure(model)
                self.assertEqual(flags, [p.requires_grad for p in model.parameters()])


if __name__ == '__main__':
    unittest.main()
