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
from torcheeg.models import LaBraMEncoder, LaBraMClassifier, LaBraMRegression, LaBraMEmbedding
from torcheeg.trainers import ClassifierTrainer, RegressorTrainer


def synthetic_record(record, sampling_rate, **kwargs):
    if sampling_rate != 200:
        raise ValueError('The source must use explicit 200 Hz native windows.')
    return {'eeg_data': np.random.RandomState(42).normal(size=(19, 3200)),
            'label': record[1], 'record': record}


class TestLaBraMHeadExamples(unittest.TestCase):
    def test_classifier_trains_head_preserves_backbone_and_saves_complete_state(self):
        self.run_example('classifier', LaBraMClassifier)

    def test_regression_uses_published_continuous_subject_targets_and_scalar_output(self):
        self.run_example('regression', LaBraMRegression)

    def test_embedding_preserves_feature_row_metadata_and_complete_state(self):
        self.run_example('embeddings', LaBraMEmbedding)

    def run_example(self, kind, model_type):
        example = Path(__file__).resolve().parents[2] / 'examples' / f'examples_eegmat_labram_{kind}.py'
        captured = {}

        def observe(backbone, **kwargs):
            model = model_type(backbone, **kwargs)
            captured['model'] = model
            captured['backbone'] = {key: value.clone() for key, value in backbone.state_dict().items()}
            if kind != 'embeddings':
                captured['head'] = model.task_head.output.weight.detach().clone()
            return model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            subjects = 3 if kind == 'regression' else 2
            for subject in range(subjects):
                for condition in (1, 2):
                    (source / f'Subject{subject:02d}_{condition}.edf').write_bytes(b'synthetic source placeholder')
            (source / 'subject-info.csv').write_text('Subject,Number of subtractions\nSubject00,20\nSubject01,30\nSubject02,40\n')
            output = root / 'model.pt'
            args = [str(example), '--root-path', str(source), '--cache-path', str(root / 'cache'),
                    '--batch-size', '3', '--output-path', str(output)]
            if kind != 'embeddings':
                args.extend(['--epochs', '1'])
            text = io.StringIO()
            with patch.object(MentalArithmeticDataset, 'read_record', side_effect=synthetic_record), \
                    patch(f'torcheeg.models.{model_type.__name__}', side_effect=observe), \
                    patch('sys.argv', args), redirect_stdout(text):
                namespace = runpy.run_path(str(example), run_name='__main__')
            result = ast.literal_eval(text.getvalue().strip().splitlines()[-1])
            payload = torch.load(output, weights_only=True)
            self.assertFalse(payload['checkpoint_loaded'])
            self.assertEqual(payload['sampling_rate'], 200)
            self.assertEqual(payload['native_sampling_rate'], 200)
            self.assertEqual(payload['task_config']['chunk_size'], 800)
            self.assertEqual(payload['patches'], 4)
            self.assertEqual(payload['task_config']['resample_mode'], 'window')
            self.assertTrue(payload['task_config']['normalize_features'])
            self.assertEqual(payload['channel_order'], payload['source_channel_order'])
            restored = model_type(LaBraMEncoder(**payload['backbone_config']), **payload['task_config']).eval()
            restored.load_state_dict(payload['state_dict'], strict=True)
            x = torch.randn(2, 19, 800)
            with torch.no_grad():
                torch.testing.assert_close(restored(x), captured['model'].eval()(x), rtol=0, atol=0)
            self.assertFalse(captured['model'].backbone.training)
            for name, value in captured['model'].backbone.state_dict().items():
                torch.testing.assert_close(value, captured['backbone'][name], rtol=0, atol=0)
            if kind == 'classifier':
                self.assertEqual((result['train_windows'], result['test_windows']), (8, 8))
                self.assertEqual(payload['held_out_subject'], 'Subject01')
                self.assertEqual(payload['train_subjects'], ['Subject00'])
                expected_loss = F.cross_entropy(payload['test_logits'], payload['test_targets']).item()
                self.assertAlmostEqual(result['metrics']['test_loss'], expected_loss, places=5)
            elif kind == 'regression':
                self.assertEqual((result['train_windows'], result['test_windows']), (8, 4))
                self.assertEqual(payload['held_out_subject'], 'Subject02')
                self.assertEqual(payload['train_subjects'], ['Subject00', 'Subject01'])
                torch.testing.assert_close(payload['test_targets'], torch.full((4,), 4., dtype=torch.float64))
                self.assertEqual(payload['test_predictions'].shape, (4, 1))
                import hashlib
                self.assertEqual(payload['metadata_sha256'], hashlib.sha256((source / 'subject-info.csv').read_bytes()).hexdigest())
                expected_mae = (payload['test_predictions'][:, 0] - payload['test_targets']).abs().mean().item()
                self.assertAlmostEqual(result['metrics']['test_mae'], expected_mae, places=5)
                self.assertAlmostEqual(result['metrics']['test_loss'], expected_mae, places=5)
                self.assertAlmostEqual(result['mae_metadata_units'], expected_mae * 10., places=4)
                for invalid in ['Subject,Number of subtractions\nSubject00,-1\n',
                                'Subject,Number of subtractions\nSubject00,2\nSubject00,3\n',
                                'Subject,Number of subtractions\nSubject00,inf\n']:
                    bad = root / 'bad.csv'
                    bad.write_text(invalid)
                    with self.assertRaises(ValueError):
                        namespace['read_scores'](bad)
            else:
                self.assertEqual(result['windows'], 16)
                self.assertEqual(payload['embeddings'].shape, (16, 19 * 4 * 100))
                self.assertTrue(torch.isfinite(payload['embeddings']).all())
                self.assertEqual(len(payload['metadata']), 16)
                subjects = [row['subject_id'] for row in payload['metadata']]
                self.assertEqual(subjects, ['Subject00'] * 8 + ['Subject01'] * 8)
                self.assertEqual({row['label'] for row in payload['metadata']}, {0, 1})
            if kind != 'embeddings':
                self.assertFalse(torch.equal(captured['head'], captured['model'].task_head.output.weight))


if __name__ == '__main__':
    unittest.main()
