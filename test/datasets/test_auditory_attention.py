import os
import tempfile
import unittest
from unittest.mock import patch

import mne
import numpy as np

from torcheeg.datasets import (AVEDDataset, AVEDProcessedDataset, DTUDataset,
                               DTUProcessedDataset, KULDataset,
                               KULProcessedDataset)
from torcheeg.datasets.auditory_attention import aved


class TestAuditoryAttention(unittest.TestCase):

    def test_dtu_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(DTUDataset.fake_record(**kwargs))
        item = next(DTUDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (64, 64))

    def test_dtu_processed_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(DTUProcessedDataset.fake_record(**kwargs))
        item = next(DTUProcessedDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (64, 64))

    def test_kul_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(KULDataset.fake_record(**kwargs))
        item = next(KULDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (64, 64))

    def test_kul_processed_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(KULProcessedDataset.fake_record(**kwargs))
        item = next(KULProcessedDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (64, 64))

    def test_aved_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(AVEDDataset.fake_record(**kwargs))
        item = next(AVEDDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (32, 128))

    def test_aved_processed_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(AVEDProcessedDataset.fake_record(**kwargs))
        item = next(AVEDProcessedDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (32, 128))

    def test_aved_reader_selects_named_eeg_channels_before_referencing(self):
        names = AVEDDataset.raw_channels[::-1]
        data = np.arange(len(names) * 128,
                         dtype=float).reshape(len(names), 128)
        raw = mne.io.RawArray(data, mne.create_info(
            names, 128, 'eeg'), verbose=False)
        expected = data[[names.index(name) for name in AVEDDataset.channels]]
        expected = expected - expected.mean(axis=0, keepdims=True)
        with tempfile.TemporaryDirectory() as path:
            folder = os.path.join(path, 'audio_only', 'sub1')
            os.makedirs(folder)
            for trial in range(1, 17):
                open(os.path.join(folder, f'trial{trial}.set'), 'w').close()
            with patch.object(mne.io, 'read_raw_eeglab', side_effect=lambda *args, **kwargs: raw.copy()), \
                    patch.object(aved, '_apply_notch_filter', side_effect=lambda x, *args: x), \
                    patch.object(aved, '_apply_bandpass_filter', side_effect=lambda x, *args: x), \
                    patch.object(aved, '_apply_downsample', side_effect=lambda x, *args: x):
                result = AVEDDataset.read_record(
                    record={'condition': 'audio_only', 'subject_id': 1}, root_path=path)
            self.assertEqual(len(result['samples']), 16)
            for sample in result['samples']:
                np.testing.assert_allclose(sample, expected)

    def test_aved_reader_rejects_missing_eeg_instead_of_substituting_eog(self):
        names = [name for name in AVEDDataset.raw_channels if name != 'O2']
        raw = mne.io.RawArray(np.zeros((len(names), 128)),
                              mne.create_info(names, 128, 'eeg'), verbose=False)
        with tempfile.TemporaryDirectory() as path:
            folder = os.path.join(path, 'audio_only', 'sub1')
            os.makedirs(folder)
            open(os.path.join(folder, 'trial1.set'), 'w').close()
            with patch.object(mne.io, 'read_raw_eeglab', return_value=raw), \
                    self.assertRaisesRegex(ValueError, 'O2'):
                AVEDDataset.read_record(
                    record={'condition': 'audio_only', 'subject_id': 1}, root_path=path)


if __name__ == '__main__':
    unittest.main()
