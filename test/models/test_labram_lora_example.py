import ast
from contextlib import ExitStack, redirect_stdout
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
from torcheeg.models import LaBraMEncoder, LaBraMClassifier, add_labram_lora
from torcheeg.trainers import ClassifierTrainer


def synthetic_record(record, sampling_rate, **kwargs):
    if sampling_rate != 200:
        raise ValueError('Use explicit native 200 Hz windows.')
    return dict(
        eeg_data=np.random.RandomState(42).normal(size=(19, 3200)), label=record[1], record=record
    )


class TestLaBraMLoRAExample(unittest.TestCase):
    def test_trains_registered_lora_head_preserves_all_base_state_and_restores(self):
        example = (
            Path(__file__).resolve().parents[2] / 'examples' / 'examples_eegmat_labram_lora.py'
        )
        captured = {}

        def observe(model, **kwargs):
            captured['model'] = model
            captured['native'] = {
                n: v.clone() for n, v in model.backbone.state_dict().items() if '.lora_' not in n
            }
            captured['low_rank_b'] = {
                n: p.detach().clone()
                for n, p in model.backbone.named_parameters()
                if '.lora_' in n and n.endswith('lora_B')
            }
            captured['head'] = model.task_head.output.weight.detach().clone()
            return ClassifierTrainer(model, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            for subject in range(2):
                for condition in (1, 2):
                    (source / f'Subject{subject:02d}_{condition}.edf').write_bytes(
                        b'synthetic source placeholder'
                    )
            path = root / 'model.pt'
            arguments = [
                str(example),
                '--root-path',
                str(source),
                '--cache-path',
                str(root / 'cache'),
                '--output-path',
                str(path),
                '--epochs',
                '1',
                '--batch-size',
                '3',
                '--rank',
                '4',
            ]
            text = io.StringIO()
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        MentalArithmeticDataset, 'read_record', side_effect=synthetic_record
                    )
                )
                stack.enter_context(
                    patch('torcheeg.trainers.ClassifierTrainer', side_effect=observe)
                )
                stack.enter_context(patch('sys.argv', arguments))
                stack.enter_context(redirect_stdout(text))
                runpy.run_path(str(example), run_name='__main__')
            payload = torch.load(path, weights_only=True)
            result = ast.literal_eval(text.getvalue().strip().splitlines()[-1])
            self.assertEqual((result['train_windows'], result['test_windows']), (8, 8))
            self.assertEqual(payload['held_out_subject'], 'Subject01')
            self.assertEqual(payload['train_subjects'], ['Subject00'])
            self.assertFalse(payload['checkpoint_loaded'])
            self.assertFalse(payload['task_config']['freeze_encoder'])
            self.assertEqual(
                payload['lora_config'],
                dict(
                    rank=4,
                    alpha=8.0,
                    dropout=0.1,
                    blocks=None,
                    target_modules=('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'),
                    freeze_base=True,
                ),
            )
            model = captured['model']
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.assertEqual(payload['trainable_parameters'], trainable)
            self.assertEqual(result['trainable_parameters'], trainable)
            for n, v in model.backbone.state_dict().items():
                if n in captured['native']:
                    torch.testing.assert_close(v, captured['native'][n], rtol=0, atol=0)
            params = dict(model.backbone.named_parameters())
            for n, before in captured['low_rank_b'].items():
                self.assertFalse(torch.equal(before, params[n]), n)
            self.assertTrue(all(p.requires_grad == ('.lora_' in n) for n, p in params.items()))
            self.assertFalse(torch.equal(model.task_head.output.weight, captured['head']))
            restored = LaBraMClassifier(
                LaBraMEncoder(**payload['backbone_config']), **payload['task_config']
            )
            add_labram_lora(restored, **payload['lora_config'])
            restored.load_state_dict(payload['state_dict'], strict=True)
            x = torch.randn(2, 19, 800, requires_grad=True)
            torch.testing.assert_close(restored.eval()(x), model.eval()(x), rtol=0, atol=0)
            restored.train()
            self.assertTrue(restored.backbone.training)
            self.assertTrue(restored.backbone.blocks[0].attn.qkv.training)
            restored.eval()(x).square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().max().item(), 0.0)
            self.assertTrue(
                all(
                    p.grad is None
                    for n, p in restored.backbone.named_parameters()
                    if '.lora_' not in n
                )
            )
            expected = F.cross_entropy(payload['test_logits'], payload['test_targets']).item()
            self.assertAlmostEqual(payload['metrics']['test_loss'], expected, places=5)
            self.assertAlmostEqual(
                payload['metrics']['test_accuracy'],
                (payload['test_logits'].argmax(1) == payload['test_targets']).float().mean().item(),
                places=6,
            )


if __name__ == '__main__':
    unittest.main()
