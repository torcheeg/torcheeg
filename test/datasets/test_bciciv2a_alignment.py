"""Regression tests for actual MATLAB-vector and one-based onset semantics."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from torcheeg.datasets import BCICIV2aDataset


def record(onsets=(1, 7), labels=(1, 4), artifacts=(0, 0), layout='column'):
    values = np.arange(12 * 25, dtype=np.float64).reshape(12, 25)

    def vector(items):
        array = np.asarray(items)
        return (
            array.reshape(-1, 1)
            if layout == 'column'
            else array.reshape(1, -1)
            if layout == 'row'
            else array
        )

    inner = np.empty((1, 1), dtype=object)
    inner[0, 0] = [
        values,
        vector(onsets),
        vector(labels),
        250,
        ['left', 'tongue'],
        vector(artifacts),
    ]
    data = np.empty((1, 1), dtype=object)
    data[0, 0] = inner
    return values, data


class TestBCICIV2aAlignment(unittest.TestCase):
    def test_first_last_sample_overlap_and_next_trial_boundary(self):
        values, data = record()
        clips = list(
            BCICIV2aDataset.process_record('A01T.mat', data, chunk_size=4, overlap=2)
        )
        self.assertEqual([c['info']['start_at'] for c in clips], [0, 2, 6, 8])
        self.assertEqual([c['info']['end_at'] for c in clips], [4, 6, 10, 12])
        self.assertEqual([c['info']['label'] for c in clips], [1, 1, 4, 4])
        self.assertEqual([c['info']['trial_id'] for c in clips], [0, 0, 1, 1])
        for clip, start in zip(clips, [0, 2, 6, 8]):
            np.testing.assert_array_equal(clip['eeg'], values[start : start + 4, :22].T)
        self.assertEqual(clips[-1]['eeg'][21, -1], values[-1, 21])

    def test_row_column_flat_vectors_and_artifact_skip(self):
        for layout in ['row', 'column', 'flat']:
            values, data = record(artifacts=(1, 0), layout=layout)
            clips = list(
                BCICIV2aDataset.process_record(
                    'A02E.mat', data, chunk_size=6, skip_trial_with_artifacts=True
                )
            )
            self.assertEqual(len(clips), 1)
            self.assertEqual(clips[0]['info']['start_at'], 6)
            self.assertEqual(clips[0]['info']['label'], 4)
            self.assertEqual(clips[0]['info']['subject_id'], 'A02')
            self.assertEqual(clips[0]['info']['session'], 'E')
            np.testing.assert_array_equal(clips[0]['eeg'], values[6:12, :22].T)

    def test_offset_is_applied_after_matlab_conversion_and_transforms(self):
        values, data = record()
        observed = []

        def before(values):
            observed.append(values.copy())
            return values + 10

        def transform(eeg):
            return {'eeg': eeg * 2}

        clips = list(
            BCICIV2aDataset.process_record(
                'A01T.mat',
                data,
                offset=2,
                chunk_size=4,
                before_trial=before,
                offline_transform=transform,
            )
        )
        self.assertEqual(len(observed), 1)
        np.testing.assert_array_equal(observed[0], values.T)
        self.assertEqual([c['info']['start_at'] for c in clips], [2, 8])
        for clip, start in zip(clips, [2, 8]):
            np.testing.assert_array_equal(
                clip['eeg'], (values[start : start + 4, :22].T + 10) * 2
            )

    def test_scipy_mat_round_trip_and_empty_calibration_runs(self):
        values = np.arange(12 * 25).reshape(12, 25)
        payload = np.empty((1, 2), dtype=object)
        for index, onsets in enumerate([[], [1, 7]]):
            payload[0, index] = dict(
                X=values,
                trial=np.asarray(onsets).reshape(-1, 1),
                y=np.asarray([] if index == 0 else [1, 4]).reshape(-1, 1),
                fs=250,
                classes=['left', 'tongue'],
                artifacts=np.zeros((len(onsets), 1)),
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'A01T.mat'
            savemat(path, {'data': payload})
            kwargs = BCICIV2aDataset.read_record(str(path))
            clips = list(
                BCICIV2aDataset.process_record(str(path), chunk_size=6, **kwargs)
            )
        self.assertEqual(len(clips), 2)
        self.assertEqual([c['info']['run'] for c in clips], [1, 1])
        np.testing.assert_array_equal(clips[0]['eeg'], values[:6, :22].T)
        np.testing.assert_array_equal(clips[1]['eeg'], values[6:, :22].T)

    def test_invalid_onsets_and_metadata_lengths_rejected(self):
        for onsets in [(0, 7), (7, 1), (1, 1), (1, 13), (1, 7.5), (1, float('nan'))]:
            _, data = record(onsets=onsets)
            with self.assertRaisesRegex(ValueError, 'one-based'):
                list(BCICIV2aDataset.process_record('A01T.mat', data, chunk_size=4))
        _, data = record(labels=(1,))
        with self.assertRaisesRegex(ValueError, 'matching lengths'):
            list(BCICIV2aDataset.process_record('A01T.mat', data, chunk_size=4))


if __name__ == '__main__':
    unittest.main()
