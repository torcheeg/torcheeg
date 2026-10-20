import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.models import EEGMamba


EXAMPLE = Path(__file__).resolve().parents[2] / 'examples/examples_eegmat_eegmamba.py'


def synthetic_record(record, sampling_rate, **kwargs):
    if sampling_rate != 200:
        raise ValueError('Expected200Hz.')
    samples = 1600 if record[1] == 0 else 800
    return dict(eeg_data=np.random.RandomState(42 + record[1]).normal(size=(19, samples)),
                label=record[1], record=record)


class TestEEGMambaExample(unittest.TestCase):
    def test_scaling_and_patch_order(self):
        namespace = runpy.run_path(str(EXAMPLE))
        eeg = np.arange(19*800, dtype=np.float64).reshape(19, 800)
        expected = (eeg/100).astype(np.float32).reshape(19, 4, 200)
        np.testing.assert_array_equal(namespace['prepare_window'](eeg), expected)
        for invalid in [np.zeros((18, 800)), np.full((19, 800), np.nan)]:
            with self.assertRaises(ValueError): namespace['prepare_window'](invalid)

    def test_complete_canonical_generated_checkpoint_workflow(self):
        # Full canonical architecture with generated weights; real official
        # checkpoint and EDF training are verified separately.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            for subject in range(2):
                for condition in (1, 2):
                    (source/f'Subject{subject:02d}_{condition}.edf').write_bytes(b'synthetic placeholder')
            state = EEGMamba().state_dict()
            checkpoint, artifact = root/'EEGMamba.pth', root/'probe.pt'
            torch.save(state,checkpoint)
            args = [str(EXAMPLE),'--root-path',str(source),'--checkpoint',str(checkpoint),
                    '--cache-path',str(root/'cache'),'--output-path',str(artifact),'--epochs','1','--batch-size','2']
            with patch.object(MentalArithmeticDataset,'read_record',side_effect=synthetic_record),patch('sys.argv',args):
                runpy.run_path(str(EXAMPLE),run_name='__main__')
            payload=torch.load(artifact,weights_only=True)
            metadata=json.loads(payload['metadata_json'])
            self.assertEqual((metadata['train_windows'],metadata['test_windows'],metadata['updates']),(3,3,2))
            self.assertEqual(metadata['train_subjects'],['Subject00'])
            self.assertEqual(metadata['held_out_subject'],'Subject01')
            self.assertEqual(payload['test_logits'].shape,(3,2))
            self.assertEqual(json.loads(payload['backbone_config_json'])['num_layers'],12)
            loss=torch.nn.functional.cross_entropy(payload['test_logits'],payload['test_targets']).item()
            accuracy=(payload['test_logits'].argmax(1)==payload['test_targets']).float().mean().item()
            self.assertEqual(metadata['metrics'],dict(loss=loss,accuracy=accuracy))
            for name,value in state.items():
                current=payload['state_dict']['backbone.'+name]
                torch.testing.assert_close(current,value,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
