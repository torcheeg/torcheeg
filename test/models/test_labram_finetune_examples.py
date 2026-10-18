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
from torcheeg.models import LaBraMEncoder, LaBraMClassifier
from torcheeg.trainers import ClassifierTrainer


def synthetic_record(record, sampling_rate, **kwargs):
    if sampling_rate != 200:
        raise ValueError('Use explicit native 200 Hz windows.')
    return dict(
        eeg_data=np.random.RandomState(42).normal(size=(19, 3200)), label=record[1], record=record
    )


class TestLaBraMFineTuneExamples(unittest.TestCase):
    def test_full_fine_tuning_updates_native_encoder_and_head_and_restores(self):
        self.run_example('full')

    def test_head_only_fine_tuning_preserves_native_state_and_records_exact_policy(self):
        self.run_example('head')

    def run_example(self, strategy):
        name = 'full_finetune' if strategy == 'full' else 'classifier'
        example = (
            Path(__file__).resolve().parents[2] / 'examples' / f'examples_eegmat_labram_{name}.py'
        )
        captured = {}

        def observe(model, **kwargs):
            captured['model'] = model
            captured['before'] = {n: t.clone() for n, t in model.state_dict().items()}
            self.assertEqual(model.freeze_encoder, strategy == 'head')
            model.train()
            self.assertEqual(model.backbone.training, strategy == 'full')
            for n, p in model.named_parameters():
                self.assertEqual(
                    p.requires_grad, strategy == 'full' or n.startswith('task_head.'), n
                )
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
            output = root / 'model.pt'
            arguments = [
                str(example),
                '--root-path',
                str(source),
                '--cache-path',
                str(root / 'cache'),
                '--output-path',
                str(output),
                '--epochs',
                '1',
                '--batch-size',
                '3',
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
            payload = torch.load(output, weights_only=True)
            result = ast.literal_eval(text.getvalue().strip().splitlines()[-1])
            self.assertEqual((result['train_windows'], result['test_windows']), (8, 8))
            self.assertEqual(payload['fine_tuning'], strategy)
            self.assertEqual(payload['learning_rate'], 0.0001 if strategy == 'full' else 0.001)
            self.assertEqual(payload['task_config']['freeze_encoder'], strategy == 'head')
            self.assertEqual(payload['held_out_subject'], 'Subject01')
            self.assertEqual(payload['train_subjects'], ['Subject00'])
            model = captured['model']
            names = [n for n, p in model.named_parameters() if p.requires_grad]
            self.assertEqual(payload['trainable_parameter_names'], names)
            self.assertEqual(
                payload['trainable_parameters'],
                sum(p.numel() for p in model.parameters() if p.requires_grad),
            )
            for n, tensor in model.backbone.state_dict().items():
                before = captured['before']['backbone.' + n]
                if strategy == 'full' and not n.endswith('.attn.k_norm.bias'):
                    self.assertFalse(torch.equal(tensor, before), n)
                elif strategy == 'head':
                    torch.testing.assert_close(tensor, before, rtol=0, atol=0)
                # A shared key-normalization offset cancels in softmax.
                # Its gradient can be exactly zero despite full trainability.
            for n, tensor in model.task_head.state_dict().items():
                self.assertFalse(torch.equal(tensor, captured['before']['task_head.' + n]), n)
            restored = LaBraMClassifier(
                LaBraMEncoder(**payload['backbone_config']), **payload['task_config']
            )
            restored.load_state_dict(payload['state_dict'], strict=True)
            self.assertEqual([n for n, p in restored.named_parameters() if p.requires_grad], names)
            restored.train()
            self.assertEqual(restored.backbone.training, strategy == 'full')
            self.assertTrue(restored.task_head.training)
            x = torch.randn(2, 19, 800, requires_grad=True)
            torch.testing.assert_close(restored.eval()(x), model.eval()(x), rtol=0, atol=0)
            logits = restored(x)
            (logits * torch.linspace(-0.7, 1.1, logits.numel()).reshape_as(logits)).sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().max().item(), 0.0)
            for n, p in restored.backbone.named_parameters():
                if strategy == 'head':
                    self.assertIsNone(p.grad, n)
                else:
                    self.assertIsNotNone(p.grad, n)
                    self.assertTrue(torch.isfinite(p.grad).all(), n)
            expected = F.cross_entropy(payload['test_logits'], payload['test_targets']).item()
            self.assertAlmostEqual(payload['metrics']['test_loss'], expected, places=5)
            self.assertAlmostEqual(
                payload['metrics']['test_accuracy'],
                (payload['test_logits'].argmax(1) == payload['test_targets']).float().mean().item(),
                places=6,
            )


if __name__ == '__main__':
    unittest.main()
