import tempfile
import unittest
from pathlib import Path
from test.datasets.test_base_dataset import _assert_memory_dataset

import numpy as np
import pandas as pd

from torcheeg.datasets import (StrokePatientsMIDataset,
                               StrokePatientsMIProcessedDataset)


class TestStrokes(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_stroke_patients_mi_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(StrokePatientsMIDataset.fake_record(**kwargs))
        item = next(StrokePatientsMIDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (33, 1000))

        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(StrokePatientsMIProcessedDataset.fake_record(**kwargs))
        item = next(StrokePatientsMIProcessedDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (30, 1000))


class TestFakeRecordMetadata(unittest.TestCase):
    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_fake_samples_have_consistent_keys_and_metadata(self):
        for dataset_class in (StrokePatientsMIDataset, StrokePatientsMIProcessedDataset,):
            with self.subTest(dataset=dataset_class.__name__):
                kwargs = {'record': 'fake_record'}
                kwargs.update(dataset_class.fake_record(**kwargs))
                sample = next(item for item in dataset_class.process_record(**kwargs)
                              if 'info' in item)
                self.assertIsInstance(sample['key'], str)
                self.assertEqual(sample['info']['clip_id'], sample['key'])
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


if __name__ == '__main__':
    unittest.main()
