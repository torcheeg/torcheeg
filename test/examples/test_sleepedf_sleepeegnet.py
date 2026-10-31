"""Physical contexts, source-only scaling and EEG-only full decoder evaluation."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from torcheeg.models import SleepEEGNet
from test.examples.test_sleepedf_sleeptransformer import _ScoredEpochs

EXAMPLE = (
    Path(__file__).resolve().parents[2] / 'examples/examples_sleepedf_sleepeegnet.py'
)
spec = importlib.util.spec_from_file_location('sleepedf_sleepeegnet_example', EXAMPLE)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class TestSleepEDFSleepEEGNet(unittest.TestCase):
    def test_voltage_conversion_labels_physical_contexts_and_gaps(self):
        fixture = _ScoredEpochs()
        features, labels, identities = workflow.select_contexts(fixture, 2, 10)
        self.assertEqual(tuple(features.shape), (6, 10, 1, 3000))
        expected = np.stack(
            [fixture.read_eeg('SC400', str(i)) * 1e6 for i in range(10)]
        ).astype(np.float32)
        np.testing.assert_array_equal(features[0].numpy(), expected)
        self.assertEqual(labels[0].tolist(), [0, 1, 2, 3, 4] * 2)
        self.assertEqual([row['onsets'][0] for row in identities[:2]], [0.0, 900.0])
        mask = (fixture.info.subject_id == 'SC400') & (
            fixture.info.clip_id.astype(int) >= 5
        )
        fixture.info.loc[mask, 'start_at'] += 30
        _, _, identities = workflow.select_contexts(fixture, 4, 10)
        for row in identities:
            np.testing.assert_array_equal(np.diff(row['onsets']), np.full(9, 30))
        self.assertEqual(sum(row['subject'] == 'SC400' for row in identities), 3)
        with self.assertRaises(ValueError):
            workflow.select_contexts(fixture, 1, 9)

    def test_source_only_channel_statistics(self):
        train = torch.randn(3, 10, 1, 3000)
        held_out = torch.full((2, 10, 1, 3000), 10000.0)
        scaled, test, mean, scale = workflow.standardize_from_train(train, held_out)
        torch.testing.assert_close(
            mean, train.mean((0, 1, 3), keepdim=True), rtol=0, atol=0
        )
        torch.testing.assert_close(
            scale, train.std((0, 1, 3), unbiased=False, keepdim=True), rtol=0, atol=0
        )
        torch.testing.assert_close(test, (held_out - mean) / scale, rtol=0, atol=0)
        torch.testing.assert_close(scaled.mean(), torch.tensor(0.0), rtol=0, atol=1e-6)

    def test_missing_eod_and_reserved_tokens_count_as_incorrect_stages(self):
        tokens = torch.tensor([[0, 6, 2, 3], [5, 4, 1, 2]])
        lengths = torch.tensor([2, 4])
        predictions = workflow.align_predictions(tokens, lengths, 5)
        self.assertEqual(predictions.tolist(), [[0, -1, -1, -1, -1], [-1, 4, 1, 2, -1]])
        targets = torch.tensor([[0, 0, 2, 3, 4], [0, 4, 1, 2, 0]])
        self.assertAlmostEqual(
            float((predictions == targets).float().mean()), 0.4, places=6
        )

    def test_full_teacher_training_and_label_free_autoregressive_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sleep-cassette').mkdir()
            (root / 'sleep-cassette/fixture.edf').write_bytes(
                b'synthetic scored-reader fixture'
            )
            output = root / 'model.pt'
            argv = [
                str(EXAMPLE),
                '--root-path',
                str(root),
                '--output-path',
                str(output),
                '--epochs',
                '1',
                '--max-contexts',
                '1',
                '--batch-size',
                '2',
            ]
            with patch.object(sys, 'argv', argv), patch.object(
                workflow, 'SleepEDFxDataset', return_value=_ScoredEpochs()
            ), contextlib.redirect_stdout(io.StringIO()):
                workflow.main()
            checkpoint = torch.load(output, map_location='cpu', weights_only=True)
            metadata = json.loads(checkpoint['metadata_json'])
            self.assertEqual(metadata['train_subjects'], ['SC400', 'SC401'])
            self.assertEqual(metadata['held_out_subject'], 'SC402')
            self.assertEqual(
                (
                    metadata['train_contexts'],
                    metadata['test_contexts'],
                    metadata['updates'],
                ),
                (2, 1, 1),
            )
            model = SleepEEGNet(**json.loads(checkpoint['config_json'])).eval()
            self.assertEqual(len(checkpoint['model']), 32)
            model.load_state_dict(checkpoint['model'], strict=True)
            test = (checkpoint['test_features'] - checkpoint['mean']) / checkpoint[
                'scale'
            ]
            # No test targets are passed into either autoregressive call.
            result = model.generate(test)
            torch.testing.assert_close(
                result['logits'], checkpoint['test_logits'], rtol=0, atol=0
            )
            self.assertTrue(torch.equal(result['tokens'], checkpoint['test_tokens']))
            predictions = workflow.align_predictions(
                result['tokens'], result['lengths'], 10
            )
            self.assertTrue(torch.equal(predictions, checkpoint['test_predictions']))
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                workflow.save_checkpoint({}, output)
            self.assertEqual(original, output.read_bytes())


if __name__ == '__main__':
    unittest.main()
