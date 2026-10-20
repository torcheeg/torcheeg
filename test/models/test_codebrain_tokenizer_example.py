import copy
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.models import CodeBrainTokenizer
from test.models.test_codebrain_tokenizer import TinyTokenizer


EXAMPLE = Path(__file__).resolve().parents[2] / 'examples/examples_eegmat_codebrain_tokenizer.py'


def synthetic_record(record, sampling_rate, **kwargs):
    if sampling_rate != 200:
        raise ValueError('Expected200Hz.')
    samples = 1600 if record[1] == 0 else 800
    return dict(eeg_data=np.random.RandomState(42+record[1]).normal(size=(19, samples)), label=record[1], record=record)


class TestCodeBrainTokenizerExample(unittest.TestCase):
    def test_scaling_and_patch_order(self):
        namespace = runpy.run_path(str(EXAMPLE))
        values = np.arange(19*800, dtype=np.float64).reshape(19, 800)
        expected = (values/100).astype(np.float32).reshape(19, 4, 200)
        np.testing.assert_array_equal(namespace['prepare_window'](values), expected)
        with self.assertRaises(ValueError): namespace['prepare_window'](values[:18])

    def test_workflow_full_state_rng_optimizer_and_next_step_restore(self):
        # Complete training flow with a small strictly loaded fixture. The
        # canonical325tensor pretrained model also runs separately on real EDF.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            for subject in range(2):
                for condition in (1, 2):
                    (source/f'Subject{subject:02d}_{condition}.edf').write_bytes(b'synthetic placeholder')
            checkpoint, artifact = root/'tokenizer.pt', root/'trained.pt'
            torch.save(TinyTokenizer(quantize_kmeans_init=False).state_dict(), checkpoint)
            factory = TinyTokenizer.from_pretrained
            args = [str(EXAMPLE), '--root-path', str(source), '--checkpoint', str(checkpoint),
                    '--cache-path', str(root/'cache'), '--output-path', str(artifact), '--epochs', '1', '--batch-size', '2']
            with patch.object(MentalArithmeticDataset, 'read_record', side_effect=synthetic_record), \
                    patch.object(CodeBrainTokenizer, 'from_pretrained', side_effect=factory), patch('sys.argv', args):
                runpy.run_path(str(EXAMPLE), run_name='__main__')
            payload = torch.load(artifact, weights_only=True)
            metadata = json.loads(payload['metadata_json'])
            self.assertEqual(metadata['train_subjects'], ['Subject00'])
            self.assertEqual(metadata['held_out_subject'], 'Subject01')
            self.assertEqual((metadata['train_windows'], metadata['test_windows'], metadata['updates']), (3, 3, 2))
            self.assertFalse(metadata['condition_labels_used_for_training'])
            self.assertFalse(metadata['usage_tracking'])
            self.assertEqual(metadata['contrastive_mode'], 'infonce')
            self.assertEqual(payload['predictions']['raw'].shape, (3, 76, 200))
            self.assertEqual(payload['predictions']['time_ids'].shape, (3, 76))
            torch.testing.assert_close(payload['predictions']['conditions'], torch.tensor([0, 0, 1]))
            self.assertTrue(payload['optimizer_tensors'])
            model = CodeBrainTokenizer(**json.loads(payload['model_config_json'])).eval()
            model.load_state_dict(payload['state_dict'], strict=True)
            with torch.no_grad():
                x = torch.randn(2, 19, 4, 200)
                first = model.get_codebook_indices(x)
                before = copy.deepcopy(model.state_dict())
                second = model.get_codebook_indices(x)
            for a, b in zip(first, second): torch.testing.assert_close(a, b, rtol=0, atol=0)
            for name, value in before.items(): torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
            results = []
            batch = torch.randn(2, 19, 4, 200)*.01
            for _ in range(2):
                # Separate loads avoid shared optimizer momentum storage.
                saved = torch.load(artifact, weights_only=True)
                restored = CodeBrainTokenizer(**json.loads(saved['model_config_json'])).train()
                restored.load_state_dict(saved['state_dict'], strict=True)
                optimizer = torch.optim.Adam([p for p in restored.parameters() if p.requires_grad], lr=.0001)
                optimizer.load_state_dict(dict(state=saved['optimizer_tensors'], param_groups=json.loads(saved['optimizer_groups_json'])))
                torch.set_rng_state(saved['torch_rng_state'])
                generator = torch.Generator().set_state(saved['train_generator_state'])
                order = torch.randperm(3, generator=generator)
                loss, _ = restored(batch); loss.backward(); optimizer.step()
                results.append((loss.detach(), copy.deepcopy(restored.state_dict()), copy.deepcopy(optimizer.state_dict()['state']), order))
            torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
            torch.testing.assert_close(results[0][3], results[1][3], rtol=0, atol=0)
            for name, value in results[0][1].items(): torch.testing.assert_close(value, results[1][1][name], rtol=0, atol=0)
            for key, state in results[0][2].items():
                for name, value in state.items(): torch.testing.assert_close(value, results[1][2][key][name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
