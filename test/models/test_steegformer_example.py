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
from torcheeg.models import STEEGFormer
from torcheeg.trainers import ClassifierTrainer


def synthetic_record(record, sampling_rate, **kwargs):
    if sampling_rate != 128:
        raise ValueError('ST-EEGFormer uses explicit 128 Hz source windows.')
    return {'eeg_data': np.random.RandomState(42).normal(size=(19, 2048)),
            'label': record[1], 'record': record}


class TestSTEEGFormerExample(unittest.TestCase):
    def test_linear_probe_trains_head_freezes_encoder_and_restores_all_state(self):
        example = Path(__file__).resolve().parents[2] / 'examples' / 'examples_eegmat_steegformer.py'
        captured = {}

        def observe(model, **kwargs):
            captured['model'] = model
            captured['encoder'] = {name: value.clone() for name, value in model.encoder.state_dict().items()}
            captured['head'] = model.classifier.weight.detach().clone()
            return ClassifierTrainer(model, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            for subject in range(2):
                for condition in (1, 2):
                    (source / f'Subject{subject:02d}_{condition}.edf').write_bytes(b'synthetic source placeholder')
            output = root / 'model.pt'
            arguments = [str(example), '--root-path', str(source), '--cache-path', str(root / 'cache'),
                         '--epochs', '1', '--batch-size', '3', '--output-path', str(output)]
            text = io.StringIO()
            with patch.object(MentalArithmeticDataset, 'read_record', side_effect=synthetic_record), \
                    patch('torcheeg.trainers.ClassifierTrainer', side_effect=observe), \
                    patch('sys.argv', arguments), redirect_stdout(text):
                namespace = runpy.run_path(str(example), run_name='__main__')
            result = ast.literal_eval(text.getvalue().strip().splitlines()[-1])
            payload = torch.load(output, weights_only=True)
            self.assertEqual((result['train_windows'], result['test_windows']), (8, 8))
            self.assertEqual(payload['train_subjects'], ['Subject00'])
            self.assertEqual(payload['held_out_subject'], 'Subject01')
            self.assertFalse(payload['checkpoint_loaded'])
            self.assertFalse(payload['feature_normalize'])
            self.assertEqual((payload['sampling_rate'], payload['chunk_size'], payload['patches']), (128, 512, 32))
            self.assertEqual(payload['normalization']['ddof'], 0)
            normalize = namespace['normalize_window']
            np.testing.assert_array_equal(normalize(np.full((2, 512), 7.)), np.zeros((2, 512)))
            wave = np.arange(2 * 512, dtype=np.float64).reshape(2, 512)
            np.testing.assert_allclose(normalize(wave).mean(-1), 0., atol=1e-15)
            np.testing.assert_allclose(normalize(wave).std(-1), 1., atol=1e-15)
            model = captured['model']
            self.assertFalse(torch.equal(model.classifier.weight, captured['head']))
            for name, value in model.encoder.state_dict().items():
                torch.testing.assert_close(value, captured['encoder'][name], rtol=0, atol=0)
            restored = namespace['LinearProbe'](STEEGFormer(**payload['encoder_config']), **payload['probe_config'])
            restored.load_state_dict(payload['state_dict'], strict=True)
            x = torch.randn(2, 19, 512, requires_grad=True)
            torch.testing.assert_close(restored.eval()(x), model.eval()(x), rtol=0, atol=0)
            restored(x).square().sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().sum().item(), 0.)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in restored.encoder.parameters()))
            restored.train()
            self.assertFalse(restored.encoder.training)
            self.assertTrue(restored.dropout.training)
            self.assertAlmostEqual(F.cross_entropy(payload['test_logits'], payload['test_targets']).item(),
                                   result['metrics']['test_loss'], places=5)


if __name__ == '__main__':
    unittest.main()
