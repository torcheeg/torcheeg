import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.models import TFMTokenizer, TFMEncoder


EXAMPLE = Path(__file__).resolve().parents[2] / 'examples/examples_eegmat_tfm.py'


def synthetic_record(record, sampling_rate, **kwargs):
    if sampling_rate != 200:
        raise ValueError('Expected200Hz.')
    samples = 1600 if record[1] == 0 else 800
    return dict(eeg_data=np.random.RandomState(42 + record[1]).normal(size=(19, samples)),
                label=record[1], record=record)


class TestTFMExample(unittest.TestCase):
    def test_scaling_and_patch_order(self):
        namespace = runpy.run_path(str(EXAMPLE))
        eeg = np.arange(19*800, dtype=np.float64).reshape(19, 800)
        expected = (eeg[:16]/(np.quantile(np.abs(eeg[:16]), .95, axis=-1, keepdims=True)+1e-8)).astype(np.float32)
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
            tokenizer, encoder = TFMTokenizer(), TFMEncoder()
            tokenizer_state, encoder_state = tokenizer.state_dict(), encoder.state_dict()
            tokenizer_file, encoder_file, artifact = root/'tokenizer.pt', root/'encoder.pt', root/'probe.pt'
            torch.save(tokenizer_state, tokenizer_file)
            torch.save(encoder_state, encoder_file)
            args = [str(EXAMPLE),'--root-path',str(source),
                    '--tokenizer-checkpoint',str(tokenizer_file),'--encoder-checkpoint',str(encoder_file),
                    '--cache-path',str(root/'cache'),'--output-path',str(artifact),'--epochs','1','--batch-size','2']
            with patch.object(MentalArithmeticDataset,'read_record',side_effect=synthetic_record),patch('sys.argv',args):
                runpy.run_path(str(EXAMPLE),run_name='__main__')
            payload=torch.load(artifact,weights_only=True)
            metadata=json.loads(payload['metadata_json'])
            self.assertEqual((metadata['train_windows'],metadata['test_windows'],metadata['updates']),(3,3,2))
            self.assertEqual(metadata['train_subjects'],['Subject00'])
            self.assertEqual(metadata['held_out_subject'],'Subject01')
            self.assertEqual(payload['test_logits'].shape,(3,2))
            self.assertEqual(json.loads(payload['tokenizer_config_json'])['reconstruction_head'],'linear')
            self.assertEqual(json.loads(payload['encoder_config_json'])['num_classes'],1)
            self.assertEqual(metadata['channel_order'],['Fp1','Fp2','F3','F4','F7','F8','T3','T4','C3','C4','T5','T6','P3','P4','O1','O2'])
            loss=torch.nn.functional.cross_entropy(payload['test_logits'],payload['test_targets']).item()
            accuracy=(payload['test_logits'].argmax(1)==payload['test_targets']).float().mean().item()
            self.assertEqual(metadata['metrics'],dict(loss=loss,accuracy=accuracy))
            for prefix, state in [('tokenizer', tokenizer_state), ('encoder', encoder_state)]:
                for name,value in state.items():
                    torch.testing.assert_close(payload['state_dict'][prefix+'.'+name],value,rtol=0,atol=0)



if __name__ == '__main__':
    unittest.main()
