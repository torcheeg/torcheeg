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
import numpy as np
import torch

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.models import EEGPT, EEGPTEmbedding


EXAMPLES = Path(__file__).resolve().parents[2] / 'examples'


def synthetic_arithmetic_record(record, **kwargs):
    return {'eeg_data': np.random.RandomState(42).normal(size=(19, 2048)),
            'label': record[1], 'record': record}


class TestEEGPTHeadExamples(unittest.TestCase):
    def source(self):
        return FakeDataset(event_list=['left_hand', 'right_hand'], n_sessions=1,
            n_runs=1, n_subjects=2, seed=42, sfreq=250, duration=40, n_events=4)

    def expected_counts(self, source):
        paradigm = LeftRightImagery(channels=['C3', 'Cz', 'C4'], fmin=4, fmax=38,
                                    tmin=0, tmax=4, resample=256)
        return {subject: len(paradigm.get_data(source, subjects=[subject])[0]) for subject in [1, 2]}

    def test_classifier_cli_trains_frozen_encoder_head(self):
        source = self.source()
        expected = self.expected_counts(source)
        example = EXAMPLES / 'examples_bciciv2b_eegpt_classifier.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = [str(example), '--download-path', str(root / 'raw'),
                '--cache-path', str(root / 'cache'), '--epochs', '1', '--batch-size', '2', '--freeze-encoder']
            output = io.StringIO()
            with patch('moabb.datasets.BNCI2014_004', return_value=source), \
                    patch.object(FakeDataset, 'download', return_value=None), \
                    patch('sys.argv', arguments), redirect_stdout(output):
                runpy.run_path(str(example), run_name='__main__')
            result = ast.literal_eval(output.getvalue().strip().splitlines()[-1])
            self.assertFalse(result['checkpoint_loaded'])
            self.assertTrue(result['encoder_frozen'])
            self.assertEqual(result['train_windows'], expected[1])
            self.assertEqual(result['test_windows'], expected[2])
            self.assertEqual(result['held_out_subject'], 2)
            self.assertTrue(0 <= result['metrics']['test_accuracy'] <= 1)

    def test_embedding_cli_saves_features_in_metadata_order(self):
        source = self.source()
        expected = self.expected_counts(source)
        example = EXAMPLES / 'examples_bciciv2b_eegpt_embeddings.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'features.pt'
            arguments = [str(example), '--download-path', str(root / 'raw'),
                '--cache-path', str(root / 'cache'), '--output-path', str(artifact), '--batch-size', '2']
            output = io.StringIO()
            with patch('moabb.datasets.BNCI2014_004', return_value=source), \
                    patch.object(FakeDataset, 'download', return_value=None), \
                    patch('sys.argv', arguments), redirect_stdout(output):
                runpy.run_path(str(example), run_name='__main__')
            result = ast.literal_eval(output.getvalue().strip().splitlines()[-1])
            payload = torch.load(artifact, weights_only=True)
            self.assertEqual(result['windows'], sum(expected.values()))
            self.assertEqual(payload['embeddings'].shape, (sum(expected.values()), 1024))
            self.assertTrue(torch.isfinite(payload['embeddings']).all())
            self.assertEqual(len(payload['metadata']), sum(expected.values()))
            self.assertEqual([row['subject_id'] for row in payload['metadata']],
                             [1] * expected[1] + [2] * expected[2])
            self.assertEqual(len({(row['_record_id'], row['clip_id']) for row in payload['metadata']}), sum(expected.values()))
            self.assertFalse(payload['checkpoint_loaded'])
            self.assertEqual(payload['sampling_rate'], 256)
            encoder = EEGPT(**payload['encoder_config'])
            encoder.load_state_dict(payload['encoder_state'], strict=True)
            model = EEGPTEmbedding(encoder).eval()
            paradigm = LeftRightImagery(channels=['C3', 'Cz', 'C4'], fmin=4, fmax=38,
                                        tmin=0, tmax=4, resample=256)
            epochs = paradigm.get_data(source, subjects=[1])[0]
            transform = transforms.Compose([transforms.MeanStdNormalize(axis=-1), transforms.ToTensor()])
            first = transform(eeg=epochs[0, :, :1024])['eeg'].unsqueeze(0)
            torch.testing.assert_close(model(first)[0], payload['embeddings'][0])


    def test_regression_cli_uses_continuous_metadata_and_subject_holdout(self):
        example = EXAMPLES / 'examples_eegmat_eegpt_regression.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'raw'
            source.mkdir()
            for subject in range(3):
                for condition in [1, 2]:
                    (source / f'Subject{subject:02d}_{condition}.edf').write_bytes(b'synthetic EDF placeholder')
            (source / 'subject-info.csv').write_text('Subject,Number of subtractions\nSubject00,10.5\nSubject01,20.5\nSubject02,30.5\n')
            arguments = [str(example), '--root-path', str(source), '--cache-path', str(root / 'cache'),
                         '--epochs', '1', '--batch-size', '2']
            output = io.StringIO()
            with patch.object(MentalArithmeticDataset, 'read_record', side_effect=synthetic_arithmetic_record), \
                    patch('sys.argv', arguments), redirect_stdout(output):
                runpy.run_path(str(example), run_name='__main__')
            result = ast.literal_eval(output.getvalue().strip().splitlines()[-1])
            self.assertEqual(result['target'], 'Number of subtractions')
            self.assertEqual(result['target_scale'], 10.0)
            self.assertEqual(result['held_out_subject'], 'Subject02')
            self.assertEqual(result['train_windows'], 4)
            self.assertEqual(result['test_windows'], 2)
            self.assertTrue(np.isfinite(result['mae_metadata_units']))
            self.assertGreaterEqual(result['mae_metadata_units'], 0)
            self.assertFalse(result['checkpoint_loaded'])

    def test_regression_metadata_rejects_ambiguous_or_nonfinite_targets(self):
        read_scores = runpy.run_path(str(EXAMPLES / 'examples_eegmat_eegpt_regression.py'))['read_scores']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scores.csv'
            path.write_text('Subject,Number of subtractions\nSubject00,9.7\nSubject01,29.35\n')
            self.assertEqual(read_scores(path), {'subject00': 9.7, 'subject01': 29.35})
            for contents in ['wrong,columns\n1,2\n', 'Subject,Number of subtractions\n',
                             'Subject,Number of subtractions\nSubject00,1\nsubject00,2\n',
                             'Subject,Number of subtractions\nSubject00,inf\n',
                             'Subject,Number of subtractions\nSubject00,-1\n',
                             'Subject,Number of subtractions\nmissing,1\n']:
                with self.subTest(contents=contents):
                    path.write_text(contents)
                    with self.assertRaises(ValueError):
                        read_scores(path)


if __name__ == '__main__':
    unittest.main()
