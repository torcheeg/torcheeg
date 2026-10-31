"""Complete annotation intervals and safe exports, with real optional YASA."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mne
import numpy as np

from torcheeg.models import YASASleepStaging

EXAMPLE = Path(__file__).resolve().parents[2] / 'examples/examples_sleepedf_yasa.py'
spec = importlib.util.spec_from_file_location('sleepedf_yasa_example', EXAMPLE)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class TestSleepEDFYASAExample(unittest.TestCase):
    def test_physical_complete_intervals_merge_stages_and_preserve_gaps(self):
        annotations = mne.Annotations(
            [0, 30, 60, 90, 120, 150],
            [30, 30, 30, 30, 30, 45],
            [
                'Sleep stage W',
                'Sleep stage 1',
                'Sleep stage 3',
                'Sleep stage 4',
                'Movement time',
                'Sleep stage R',
            ],
        )
        onsets = np.arange(7) * 30.0
        np.testing.assert_array_equal(
            workflow.annotation_labels(annotations, onsets), [0, 1, 3, 3, -1, 4, -1]
        )
        np.testing.assert_array_equal(
            workflow.annotation_labels(annotations, np.array([60.0, 90.0, 120.0])),
            [3, 3, -1],
        )
        conflict = mne.Annotations([0, 0], [30, 30], ['Sleep stage W', 'Sleep stage 1'])
        with self.assertRaises(ValueError):
            workflow.annotation_labels(conflict, np.array([0.0]))

    def test_atomic_safe_numeric_npz_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.npz'
            payload = dict(
                labels=np.array([0, 1]),
                probabilities=np.eye(5)[:2],
                metadata_json=np.array('{"version":1}'),
            )
            workflow.save_result(payload, output)
            original = output.read_bytes()
            with np.load(output, allow_pickle=False) as result:
                np.testing.assert_array_equal(result['labels'], [0, 1])
                self.assertEqual(
                    json.loads(str(result['metadata_json'])), {'version': 1}
                )
            with self.assertRaises(FileExistsError):
                workflow.save_result({}, output)
            self.assertEqual(original, output.read_bytes())

    @unittest.skipUnless(
        importlib.util.find_spec('yasa'),
        'Optional full pretrained YASA backend not installed.',
    )
    def test_full_pretrained_workflow_accesses_annotations_only_after_prediction(self):
        rng = np.random.default_rng(17)
        raw = mne.io.RawArray(
            rng.normal(size=(1, 30000)) * 10e-6,
            mne.create_info(['EEG Fpz-Cz'], 100, 'eeg'),
            verbose=False,
        )
        annotations = mne.Annotations(
            [0, 60, 90, 180, 240],
            [60, 30, 90, 60, 60],
            [
                'Sleep stage W',
                'Sleep stage 1',
                'Sleep stage 2',
                'Sleep stage 3',
                'Sleep stage R',
            ],
        )
        original_waveform = raw.get_data().copy()
        events = []
        original_predict = YASASleepStaging.predict

        def complete_predict(model, recording):
            self.assertEqual(len(recording.annotations), 0)
            result = original_predict(model, recording)
            events.append('prediction_complete')
            return result

        def read_annotations(*args, **kwargs):
            self.assertEqual(events, ['prediction_complete'])
            events.append('annotations_read')
            return annotations

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            psg = root / 'SC4001E0-PSG.edf'
            scoring = root / 'SC4001EC-Hypnogram.edf'
            psg.write_bytes(b'synthetic continuous reader fixture')
            scoring.write_bytes(b'synthetic scoring reader fixture')
            output = root / 'result.npz'
            argv = [
                str(EXAMPLE),
                '--psg-path',
                str(psg),
                '--hypnogram-path',
                str(scoring),
                '--output-path',
                str(output),
            ]
            with patch.object(sys, 'argv', argv), patch.object(
                workflow.mne.io, 'read_raw_edf', return_value=raw
            ), patch.object(
                workflow.mne, 'read_annotations', side_effect=read_annotations
            ), patch.object(
                YASASleepStaging, 'predict', autospec=True, side_effect=complete_predict
            ), contextlib.redirect_stdout(io.StringIO()):
                workflow.main()
            self.assertEqual(events, ['prediction_complete', 'annotations_read'])
            np.testing.assert_array_equal(raw.get_data(), original_waveform)
            with np.load(output, allow_pickle=False) as result:
                meta = json.loads(str(result['metadata_json']))
                self.assertEqual(meta['reference_stage_counts'], [2, 1, 3, 2, 2])
                self.assertEqual(
                    (meta['scored_epochs'], meta['total_prediction_epochs']), (10, 10)
                )
                self.assertEqual(
                    meta['annotation_access'],
                    'evaluation only, after EEG-only prediction',
                )
                np.testing.assert_array_equal(
                    result['scoring_mask'], np.ones(10, dtype=bool)
                )
                np.testing.assert_allclose(
                    result['probabilities'].sum(1), np.ones(10), rtol=0, atol=1e-6
                )


if __name__ == '__main__':
    unittest.main()
