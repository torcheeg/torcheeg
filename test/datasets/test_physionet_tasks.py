import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from torcheeg.datasets import PhysioNetMIDataset
from test.datasets.test_physionet_mi import SyntheticPhysioNetMI


class TestPhysioNetTasks(unittest.TestCase):
    def test_exact_task_label_maps_and_exclusion(self):
        for task, expected in [
            ('four_class', {'04': [0, 1], '06': [2, 3]}),
            ('left_right', {'04': [0, 1], '06': []}),
            ('hands_feet', {'04': [0, 0], '06': [0, 1]}),
        ]:
            for run, labels in expected.items():
                with self.subTest(task=task, run=run):
                    data = np.stack([np.full((64, 800), 1.),
                                     np.full((64, 800), 2.)])
                    rows = list(PhysioNetMIDataset.process_record(
                        record=f'S001/S001R{run}.edf', epochs_data=data,
                        events=np.array([2, 3]), task_id=run, task=task,
                        chunk_size=800))
                    self.assertEqual([row['info']['label'] for row in rows], labels)
                    self.assertTrue(all(row['info']['task'] == task for row in rows))
                    for index, row in enumerate(rows):
                        np.testing.assert_array_equal(row['eeg'], data[index])

    def source(self, root):
        path = Path(root) / 'source' / 'S001'
        path.mkdir(parents=True)
        for run in ['04', '06', '08', '10', '12', '14']:
            (path / f'S001R{run}.edf').touch()
        return str(path.parent)

    def test_discovery_selects_runs_before_preprocessing(self):
        dataset = PhysioNetMIDataset.__new__(PhysioNetMIDataset)
        with tempfile.TemporaryDirectory() as root:
            source = self.source(root)
            records = dataset.set_records(source, task='left_right')
            self.assertEqual([Path(record).name for record in records],
                             ['S001R04.edf', 'S001R08.edf', 'S001R12.edf'])
            self.assertEqual(len(dataset.set_records(source, task='hands_feet')), 6)

    def test_task_validation_and_class_count(self):
        for task in ['unknown', None, [], 1]:
            with self.subTest(task=task), self.assertRaises(ValueError):
                PhysioNetMIDataset(task=task, verbose=False)
        with tempfile.TemporaryDirectory() as root:
            source = self.source(root)
            for task, count, classes in [('four_class', 12, 4),
                                          ('left_right', 6, 2),
                                          ('hands_feet', 12, 2)]:
                dataset = SyntheticPhysioNetMI(root_path=source, task=task,
                    chunk_size=800, io_mode='memory', verbose=False)
                self.assertEqual(dataset.num_classes, classes)
                self.assertEqual(len(dataset), count)
                self.assertEqual(set(dataset.info['label']), set(range(classes)))

    def fingerprints(self, path):
        return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(path).rglob('*') if p.is_file()}

    def test_mismatched_task_cache_is_rejected_and_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            source = self.source(root)
            cache = str(Path(root) / 'cache')
            dataset = SyntheticPhysioNetMI(root_path=source, task='left_right',
                chunk_size=800, io_path=cache, io_mode='pickle', verbose=False)
            before = self.fingerprints(cache)
            with self.assertRaisesRegex(ValueError, 'cached imagery task|cache configuration'):
                SyntheticPhysioNetMI(root_path=source, task='hands_feet',
                    io_path=cache, io_mode='pickle', verbose=False)
            self.assertEqual(before, self.fingerprints(cache))
            self.assertEqual(set(dataset.info['task']), {'left_right'})

    def test_legacy_four_class_cache_compatibility(self):
        with tempfile.TemporaryDirectory() as root:
            source = self.source(root)
            cache = str(Path(root) / 'cache')
            SyntheticPhysioNetMI(root_path=source, task='four_class',
                chunk_size=800, io_path=cache, io_mode='pickle', verbose=False)
            for marker in Path(cache).rglob('.torcheeg-*.json'):
                marker.unlink()
            for path in Path(cache).glob('_record_*/info.csv'):
                pd.read_csv(path).drop(columns=['task']).to_csv(path, index=False)
            before = self.fingerprints(cache)
            dataset = SyntheticPhysioNetMI(root_path=source, io_path=cache,
                io_mode='pickle', verbose=False)
            self.assertEqual(len(dataset), 12)
            self.assertEqual(set(dataset.info['label']), {0, 1, 2, 3})
            with self.assertRaisesRegex(ValueError, 'legacy cache'):
                SyntheticPhysioNetMI(root_path=source, task='hands_feet',
                    io_path=cache, io_mode='pickle', verbose=False)
            self.assertEqual(before, self.fingerprints(cache))


if __name__ == '__main__':
    unittest.main()
