import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from moabb.datasets.fake import FakeDataset
from moabb.paradigms import LeftRightImagery
import torch
from torch.nn import functional as F

from torcheeg import transforms
from torcheeg.models import EEGPT, EEGPTPredictor, EEGPTReconstructor, EEGPTFinetuneClassifier


class TestEEGPTFinetuneExamples(unittest.TestCase):
    def test_reconstructor_cli_frozen_encoder_and_complete_saved_task(self):
        self.run_example('reconstructor', freeze=True)

    def test_predictor_cli_joint_training_and_complete_saved_task(self):
        self.run_example('predictor', freeze=False)

    def run_example(self, kind, freeze):
        source = FakeDataset(event_list=['left_hand', 'right_hand'], n_sessions=1,
            n_runs=1, n_subjects=2, seed=42, sfreq=250, duration=40, n_events=4)
        paradigm = LeftRightImagery(channels=['C3', 'Cz', 'C4'], fmin=4, fmax=38,
                                    tmin=0, tmax=4, resample=256)
        expected = {subject: len(paradigm.get_data(source, subjects=[subject])[0]) for subject in [1, 2]}
        example = Path(__file__).resolve().parents[2] / 'examples' / f'examples_bciciv2b_eegpt_finetune_{kind}.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'task.pt'
            arguments = [str(example), '--download-path', str(root / 'raw'), '--cache-path', str(root / 'cache'),
                         '--epochs', '1', '--batch-size', '2', '--output-path', str(output)]
            if freeze:
                arguments.append('--freeze-encoder')
            text = io.StringIO()
            with patch('moabb.datasets.BNCI2014_004', return_value=source), \
                    patch.object(FakeDataset, 'download', return_value=None), \
                    patch('sys.argv', arguments), redirect_stdout(text):
                runpy.run_path(str(example), run_name='__main__')
            result = ast.literal_eval(text.getvalue().strip().splitlines()[-1])
            self.assertEqual(result['held_out_subject'], 2)
            self.assertEqual(result['train_windows'], expected[1])
            self.assertEqual(result['test_windows'], expected[2])
            self.assertFalse(result['checkpoint_loaded'])
            self.assertEqual(result['encoder_frozen'], freeze)
            self.assertEqual(result['decoder'], kind)
            payload = torch.load(output, weights_only=True)
            self.assertEqual(payload['train_subjects'], [1])
            self.assertEqual(payload['test_subject'], 2)
            self.assertEqual(payload['sampling_rate'], 256)
            self.assertEqual(payload['preprocessing']['channels'], ['C3','Cz','C4'])
            encoder = EEGPT(**payload['encoder_config'])
            decoder = (EEGPTPredictor if kind == 'predictor' else EEGPTReconstructor)(**payload['decoder_config'])
            model = EEGPTFinetuneClassifier(encoder,decoder,**payload['task_config']).eval()
            model.load_state_dict(payload['task_state'], strict=True)
            samples, labels, _ = paradigm.get_data(source, subjects=[2])
            transform = transforms.Compose([transforms.MeanStdNormalize(axis=-1), transforms.ToTensor()])
            eeg = torch.stack([transform(eeg=sample[:, :1024])['eeg'] for sample in samples])
            with torch.no_grad():
                actual = model(eeg)
            torch.testing.assert_close(actual, payload['test_logits'])
            targets = torch.tensor([{'left_hand':0,'right_hand':1}[label] for label in labels])
            torch.testing.assert_close(targets, payload['test_targets'])
            accuracy = (actual.argmax(1) == targets).float().mean().item()
            loss = F.cross_entropy(actual,targets).item()
            self.assertAlmostEqual(accuracy, result['metrics']['test_accuracy'], places=6)
            self.assertAlmostEqual(loss, result['metrics']['test_loss'], places=5)


if __name__ == '__main__':
    unittest.main()
