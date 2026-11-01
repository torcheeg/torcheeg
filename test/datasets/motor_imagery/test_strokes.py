import tempfile
import unittest
from pathlib import Path
from test.datasets.test_cached_dataset import (_assert_memory_dataset,
                                               _process_fake)

import numpy as np
import pandas as pd

from torcheeg.datasets import (StrokePatientsMIDataset,
                               StrokePatientsMIProcessedDataset)


class TestStrokes(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_stroke_patients_mi_dataset(self):
        kwargs = {'record': 'fake_record'}
        kwargs.update(StrokePatientsMIDataset.data_class.fake_record(**kwargs))
        item = next(_process_fake(StrokePatientsMIDataset, **kwargs))
        self.assertEqual(item['eeg'].shape, (33, 500))
        kwargs = {'record': 'fake_record'}
        kwargs.update(
            StrokePatientsMIProcessedDataset.data_class.fake_record(**kwargs))
        item = next(_process_fake(StrokePatientsMIProcessedDataset, **kwargs))
        self.assertEqual(item['eeg'].shape, (30, 500))


class TestFakeRecordMetadata(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_fake_samples_have_consistent_keys_and_metadata(self):
        for dataset_class in (StrokePatientsMIDataset, StrokePatientsMIProcessedDataset):
            with self.subTest(dataset=dataset_class.__name__):
                kwargs = {'record': 'fake_record'}
                kwargs.update(dataset_class.data_class.fake_record(**kwargs))
                sample = next((item for item in _process_fake(
                    dataset_class, **kwargs) if 'info' in item))
                self.assertIsInstance(sample['info'], dict)
                self.assertNotIn('key', sample)
                self.assertNotIn('clip_id', sample['info'])
                self.assertGreater(sample['eeg'].size, 0)
                self.assertTrue(np.isfinite(sample['eeg']).all())

    def test_memory_construction_preserves_preprocessing_and_item_transforms(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        names = [name for name in StrokePatientsMIDataset.channels if name != 'CPz']
        pd.DataFrame({'Participant_ID': [
                     'sub-01'], 'age': [30]}).to_csv(root / 'participants.tsv', sep='\t', index=False)
        pd.DataFrame({'name': names}).to_csv(
            root / 'task-motor-imagery_electrodes.tsv', sep='\t', index=False)
        pd.DataFrame({'name': names, 'type': 'EEG'}).to_csv(
            root / 'task-motor-imagery_channels.tsv', sep='\t', index=False)
        pd.DataFrame({'value': [1, 2]}).to_csv(
            root / 'task-motor-imagery_events.tsv', sep='\t', index=False)
        for dataset_class in (StrokePatientsMIDataset, StrokePatientsMIProcessedDataset):
            with self.subTest(dataset=dataset_class.__name__):
                _assert_memory_dataset(
                    self, dataset_class, root_path=str(root))

    def test_explicit_data_and_shortcut_include_the_same_subject_metadata(self):
        from unittest.mock import patch

        from torcheeg.datasets import IODataset
        from torcheeg.io import MemoryEEGIO, MemoryInfoIO
        from torcheeg.runners import SequentialRunner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = [
                name for name in StrokePatientsMIDataset.channels if name != 'CPz']
            pd.DataFrame({'Participant_ID': [
                         'sub-01'], 'age': [30]}).to_csv(root / 'participants.tsv', sep='\t', index=False)
            pd.DataFrame({'name': names}).to_csv(
                root / 'task-motor-imagery_electrodes.tsv', sep='\t', index=False)
            pd.DataFrame({'name': names, 'type': 'EEG'}).to_csv(
                root / 'task-motor-imagery_channels.tsv', sep='\t', index=False)
            pd.DataFrame({'value': [1, 2]}).to_csv(
                root / 'task-motor-imagery_events.tsv', sep='\t', index=False)
            for dataset_class in (StrokePatientsMIDataset, StrokePatientsMIProcessedDataset):
                data_class = dataset_class.data_class
                record = data_class.fake_record()['record']
                payload = {'eeg': np.arange(
                    8000, dtype=np.float32).reshape(1, 2, 4000)}
                with self.subTest(dataset=dataset_class.__name__), patch.object(data_class, 'set_records', return_value=[record]), patch.object(data_class, 'read_record', return_value=payload):
                    eeg, info = (MemoryEEGIO(), MemoryInfoIO())
                    data = data_class(root_path=str(root))
                    IODataset(eeg, info).extend(SequentialRunner()(data))
                    explicit = IODataset(eeg, info)
                    shortcut = dataset_class(root_path=str(
                        root), io_mode='memory', verbose=False)
                self.assertEqual(len(explicit), len(shortcut))
                self.assertGreater(len(explicit), 0)
                pd.testing.assert_frame_equal(explicit.info.drop(
                    columns=['_record_id']), shortcut.info.drop(columns=['_record_id']))
                self.assertEqual(explicit[0]['info']['age'], 30)
                self.assertEqual(explicit[0]['info']
                                 ['Participant_ID'], 'sub-01')
                np.testing.assert_array_equal(
                    explicit[0]['eeg'], shortcut[0]['eeg'])
                self.assertEqual(shortcut.subjects_info.iloc[0]['age'], 30)


if __name__ == '__main__':
    unittest.main()
