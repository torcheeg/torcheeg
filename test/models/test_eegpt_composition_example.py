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
from torch import nn
from torch.nn import functional as F

from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.models import (EEGPT, EEGPTReconstructor, EEGPTFinetuneClassifier,
    EEGPTInputAdapter, EEGPTAdaptedModel, add_eegpt_lora)


def synthetic_record(record, sampling_rate, **kwargs):
    if sampling_rate != 200:
        raise ValueError('Use 200 Hz source windows.')
    return {'eeg_data':np.random.RandomState(42).normal(size=(19,1600)),
            'label':record[1],'record':record}


class TestEEGPTCompositionExample(unittest.TestCase):
    def test_complete_cli_training_and_safe_composed_state(self):
        example = Path(__file__).resolve().parents[2]/'examples'/'examples_eegmat_eegpt_composition.py'
        captured = {}
        def observe(adapter,task):
            captured['frozen'] = {n:p.detach().clone() for n,p in task.encoder.named_parameters() if not p.requires_grad}
            captured['model'] = EEGPTAdaptedModel(adapter,task)
            return captured['model']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'
            source.mkdir()
            for subject in range(2):
                for condition in (1,2):
                    (source/f'Subject{subject:02d}_{condition}.edf').write_bytes(b'synthetic source placeholder')
            artifact = root/'task.pt'
            args = [str(example),'--root-path',str(source),'--cache-path',str(root/'cache'),
                    '--epochs','1','--batch-size','2','--output-path',str(artifact)]
            output = io.StringIO()
            with patch.object(MentalArithmeticDataset,'read_record',side_effect=synthetic_record), \
                    patch('torcheeg.models.EEGPTAdaptedModel',side_effect=observe), \
                    patch('sys.argv',args),redirect_stdout(output):
                runpy.run_path(str(example),run_name='__main__')
            result = ast.literal_eval(output.getvalue().strip().splitlines()[-1])
            self.assertEqual(result['held_out_subject'],'Subject01')
            self.assertEqual((result['train_windows'],result['test_windows']),(4,4))
            self.assertEqual((result['task_input_channels'],result['encoder_slots']),(5,3))
            self.assertEqual((result['source_samples'],result['adapted_samples']),(800,1024))
            self.assertFalse(result['checkpoint_loaded'])
            payload = torch.load(artifact,weights_only=True)
            self.assertEqual(len(payload['source_channel_order']),19)
            self.assertEqual(payload['encoder_token_names'],['C3','CZ','C4'])
            self.assertEqual(payload['source_sampling_rate'],200)
            self.assertEqual(payload['target_sampling_rate'],256)
            self.assertEqual(payload['window_seconds'],4)
            task = EEGPTFinetuneClassifier(EEGPT(**payload['encoder_config']),
                EEGPTReconstructor(**payload['decoder_config']),**payload['task_config'])
            add_eegpt_lora(task,**payload['tuning_config'])
            restored = EEGPTAdaptedModel(EEGPTInputAdapter(**payload['adapter_config']),task).eval()
            restored.load_state_dict(payload['state_dict'],strict=True)
            x = torch.linspace(-1,1,2*19*800).reshape(2,19,800)
            torch.testing.assert_close(restored(x),captured['model'].eval()(x))
            accuracy = (payload['test_logits'].argmax(1)==payload['test_targets']).float().mean().item()
            loss = F.cross_entropy(payload['test_logits'],payload['test_targets']).item()
            self.assertAlmostEqual(accuracy,result['metrics']['test_accuracy'],places=6)
            self.assertAlmostEqual(loss,result['metrics']['test_loss'],places=5)
            for name,p in captured['model'].model.encoder.named_parameters():
                if name in captured['frozen']:
                    torch.testing.assert_close(p,captured['frozen'][name],rtol=0,atol=0)
            self.assertTrue(any(p.detach().abs().sum().item()>0 for n,p in captured['model'].model.encoder.named_parameters() if n.endswith('lora_B')))
            counters = [m.num_batches_tracked.item() for m in captured['model'].adapter.modules() if isinstance(m,nn.BatchNorm2d)]
            self.assertEqual(counters,[2,2])


if __name__ == '__main__':
    unittest.main()
