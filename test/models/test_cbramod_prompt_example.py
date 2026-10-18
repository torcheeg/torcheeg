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
from torcheeg.models import CBraMod, CBraModClassifier, add_cbramod_prompts


def synthetic_record(record, sampling_rate, **kwargs):
    if sampling_rate != 256:
        raise ValueError('Use 256 Hz source samples before patchwise interpolation.')
    return {'eeg_data': np.random.RandomState(42).normal(size=(19, 4096)),
            'label': record[1], 'record': record}


class TestCBraModPromptExample(unittest.TestCase):
    def test_training_preserves_base_and_restores_augmented_tensor_state(self):
        example = Path(__file__).resolve().parents[2] / 'examples' / 'examples_eegmat_cbramod_prompts.py'
        captured = {}

        def observe(model, **kwargs):
            captured['base'] = {name: value.clone() for name, value in model.backbone.state_dict().items()}
            captured['head'] = model.task_head.output.weight.detach().clone()
            result = add_cbramod_prompts(model, **kwargs)
            captured['model'] = model
            captured['tuning'] = {name: value.clone() for name, value in model.backbone.state_dict().items() if name.endswith('prompt_tokens')}
            return result

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
                    patch('torcheeg.models.add_cbramod_prompts', side_effect=observe), \
                    patch('sys.argv', arguments), redirect_stdout(text):
                runpy.run_path(str(example), run_name='__main__')
            result = ast.literal_eval(text.getvalue().strip().splitlines()[-1])
            payload = torch.load(output, weights_only=True)
            self.assertEqual((result['train_windows'], result['test_windows']), (8, 8))
            self.assertEqual(payload['train_subjects'], ['Subject00'])
            self.assertEqual(payload['held_out_subject'], 'Subject01')
            self.assertFalse(payload['checkpoint_loaded'])
            self.assertFalse(payload['task_config']['freeze_encoder'])
            self.assertEqual(payload['tuning_config'], dict(num_prompts=4, num_electrodes=None, freeze_base=True))
            self.assertEqual((payload['sampling_rate'], payload['target_patch_sampling_rate']), (256, 200))
            restored = CBraModClassifier(CBraMod(**payload['backbone_config']), **payload['task_config'])
            add_cbramod_prompts(restored, **payload['tuning_config'])
            restored.load_state_dict(payload['state_dict'], strict=True)
            inputs = torch.linspace(-1, 1, 2 * 19 * 1024).reshape(2, 19, 1024)
            with torch.no_grad():
                torch.testing.assert_close(restored.eval()(inputs), captured['model'].eval()(inputs), rtol=0, atol=0)
            for name, value in captured['model'].backbone.state_dict().items():
                if name in captured['base']:
                    torch.testing.assert_close(value, captured['base'][name], rtol=0, atol=0)
            name = 'encoder.prompt_tokens'
            self.assertFalse(torch.equal(captured['tuning'][name], captured['model'].backbone.state_dict()[name]))
            self.assertFalse(torch.equal(captured['head'], captured['model'].task_head.output.weight))
            for name, parameter in captured['model'].backbone.named_parameters():
                self.assertEqual(parameter.requires_grad, name.endswith('prompt_tokens'))
            expected_loss = F.cross_entropy(payload['test_logits'], payload['test_targets']).item()
            self.assertAlmostEqual(expected_loss, result['metrics']['test_loss'], places=5)
            restored.train()
            self.assertTrue(restored.backbone.training)
            self.assertTrue(all(block.training for block in restored.backbone.encoder.layers))


if __name__ == '__main__':
    unittest.main()
