import ast
from contextlib import redirect_stdout
import hashlib
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
from torcheeg.models import CBraMod
from torcheeg.trainers import ClassifierTrainer


def synthetic_record(record, sampling_rate, **kwargs):
    if sampling_rate != 200:
        raise ValueError('Use 200 Hz source samples.')
    return {'eeg_data': np.random.RandomState(42).normal(size=(19, 3200)),
            'label': record[1], 'record': record}


class TestCBraModExample(unittest.TestCase):
    def test_small_random_backbone_probe_cli_and_complete_restore(self):
        self.run_example(checkpoint=False)

    def test_safe_canonical_component_checkpoint_cli_and_complete_restore(self):
        # This generated local state is canonical in shape, not pretrained.
        self.run_example(checkpoint=True)

    def run_example(self, checkpoint):
        example = Path(__file__).resolve().parents[2] / 'examples' / 'examples_eegmat_cbramod.py'
        captured = {}

        def observe(model, *args, **kwargs):
            captured['model'] = model
            captured['backbone'] = {n: v.detach().clone() for n, v in model.backbone.state_dict().items()}
            captured['head'] = model.head.weight.detach().clone()
            return ClassifierTrainer(model, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            for subject in range(2):
                for condition in (1, 2):
                    (source / f'Subject{subject:02d}_{condition}.edf').write_bytes(b'synthetic source placeholder')
            artifact = root / 'probe.pt'
            arguments = [str(example), '--root-path', str(source), '--cache-path', str(root / 'cache'),
                         '--epochs', '1', '--batch-size', '3', '--output-path', str(artifact)]
            checkpoint_path = root / 'synthetic-canonical-state.pt'
            if checkpoint:
                torch.save(CBraMod().state_dict(), checkpoint_path)
                arguments.extend(['--checkpoint', str(checkpoint_path)])
            text = io.StringIO()
            with patch.object(MentalArithmeticDataset, 'read_record', side_effect=synthetic_record), \
                    patch('torcheeg.trainers.ClassifierTrainer', side_effect=observe), \
                    patch('sys.argv', arguments), redirect_stdout(text):
                namespace = runpy.run_path(str(example), run_name='__main__')
            result = ast.literal_eval(text.getvalue().strip().splitlines()[-1])
            self.assertEqual(result['held_out_subject'], 'Subject01')
            self.assertEqual((result['train_windows'], result['test_windows']), (8, 8))
            self.assertEqual(result['checkpoint_loaded'], checkpoint)
            payload = torch.load(artifact, weights_only=True)
            self.assertEqual(payload['sampling_rate'], 200)
            self.assertEqual((payload['patch_size'], payload['patches']), (200, 4))
            self.assertEqual(payload['scale_microvolts'], 100.)
            self.assertEqual(payload['filters']['bandpass_hz'], (.3, 75.))
            self.assertEqual(payload['filters']['notch_hz'], 60.)
            self.assertEqual(payload['channel_order'], ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'])
            self.assertNotEqual(payload['channel_order'], payload['source_channel_order'])
            if checkpoint:
                self.assertEqual(payload['checkpoint_sha256'], hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
            else:
                self.assertIsNone(payload['checkpoint_sha256'])
            restored = namespace['FrozenCBraModProbe'](CBraMod(**payload['backbone_config'])).eval()
            restored.load_state_dict(payload['state_dict'], strict=True)
            eeg = torch.linspace(-.1, .1, 2 * 19 * 800).reshape(2, 19, 800)
            with torch.no_grad():
                torch.testing.assert_close(restored(eeg), captured['model'].eval()(eeg))
            for name, value in captured['model'].backbone.state_dict().items():
                torch.testing.assert_close(value, captured['backbone'][name], rtol=0, atol=0)
            self.assertFalse(captured['model'].backbone.training)
            self.assertTrue(all(not p.requires_grad for p in captured['model'].backbone.parameters()))
            self.assertFalse(torch.equal(captured['model'].head.weight, captured['head']))
            self.assertEqual(payload['test_logits'].shape, (8, 2))
            self.assertTrue(torch.isfinite(payload['test_logits']).all())
            loss = F.cross_entropy(payload['test_logits'], payload['test_targets']).item()
            accuracy = (payload['test_logits'].argmax(1) == payload['test_targets']).float().mean().item()
            self.assertAlmostEqual(loss, result['metrics']['test_loss'], places=5)
            self.assertAlmostEqual(accuracy, result['metrics']['test_accuracy'], places=6)


if __name__ == '__main__':
    unittest.main()
