import os
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mne
import numpy as np

from torcheeg import transforms
from torcheeg.datasets import PhysioNetMIDataset
from torcheeg.datasets.constants import (PHYSIONET_MI_CHANNEL_LIST,
                                         PHYSIONET_MI_ADJACENCY_MATRIX,
                                         PHYSIONET_MI_EDF_CHANNEL_LIST,
                                         PHYSIONET_MI_CHANNEL_LOCATION_DICT)


class SyntheticPhysioNetMI(PhysioNetMIDataset):
    @staticmethod
    def read_record(record, **kwargs):
        task = os.path.basename(record)[-6:-4]
        return dict(epochs_data=np.ones((2, 64, 800)),
                    events=np.array([2, 3]), task_id=task)


class TestPhysioNetMI(unittest.TestCase):
    def record(self, task='04'):
        data = np.arange(3 * 64 * 800, dtype=float).reshape(3, 64, 800)
        return dict(record=f'S001/S001R{task}.edf', epochs_data=data,
                    events=np.array([1, 2, 3]), task_id=task)

    def test_all_run_labels_and_rest_exclusion(self):
        for task in ['04', '06', '08', '10', '12', '14']:
            with self.subTest(task=task):
                rows = list(PhysioNetMIDataset.process_record(**self.record(task)))
                expected = [0, 1] if task in ['04', '08', '12'] else [2, 3]
                self.assertEqual([r['info']['label'] for r in rows],
                                 [label for label in expected for _ in range(4)])
                self.assertEqual(len(set(r['key'] for r in rows)), 8)
                self.assertEqual({r['info']['session_id'] for r in rows}, {task})
                self.assertEqual({r['info']['subject_id'] for r in rows}, {'S001'})

    def test_exact_window_values_and_boundaries(self):
        record = self.record()
        rows = list(PhysioNetMIDataset.process_record(
            **record, chunk_size=300, overlap=100, num_channel=2))
        self.assertEqual(len(rows), 6)
        for index, row in enumerate(rows):
            epoch = index // 3 + 1
            start = index % 3 * 200
            np.testing.assert_array_equal(row['eeg'],
                record['epochs_data'][epoch, :2, start:start + 300])
            self.assertEqual(row['info']['start_at'], start)
            self.assertEqual(row['info']['end_at'], start + 300)

    def test_whole_epoch_and_hooks(self):
        calls = []
        def before(eeg):
            calls.append(eeg.shape)
            return eeg + 2
        def offline(eeg):
            return {'eeg': eeg * 3}
        record = self.record()
        rows = list(PhysioNetMIDataset.process_record(
            **record, chunk_size=-1, before_trial=before,
            offline_transform=offline))
        self.assertEqual(calls, [(64, 800), (64, 800)])
        np.testing.assert_array_equal(rows[0]['eeg'],
                                     (record['epochs_data'][1] + 2) * 3)
        with self.assertRaises(ValueError):
            list(PhysioNetMIDataset.process_record(
                **record, before_trial=lambda eeg: eeg[:, :10]))

    def test_parameter_validation(self):
        for kwargs in [dict(overlap=200), dict(overlap=-1), dict(overlap=1.5),
                       dict(sfreq=0), dict(sfreq=True), dict(num_channel=0),
                       dict(num_channel=65), dict(chunk_size=2.5), dict(chunk_size=801)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                list(PhysioNetMIDataset.process_record(**self.record(), **kwargs))

    def test_invalid_events_shapes_and_tasks(self):
        for update in [dict(events=np.array([1, 2])),
                       dict(events=np.array([1, 2, 4])), dict(task_id='03'),
                       dict(epochs_data=np.zeros((3, 64, 799))),
                       dict(epochs_data=np.zeros((3, 64)))]:
            record = self.record()
            record.update(update)
            with self.subTest(update=list(update)), self.assertRaises(ValueError):
                list(PhysioNetMIDataset.process_record(**record))

    def test_discovery_filters_and_sorts_runs(self):
        dataset = PhysioNetMIDataset.__new__(PhysioNetMIDataset)
        with tempfile.TemporaryDirectory() as root:
            for subject in ['S002', 'S001', '.hidden', 'notes']:
                os.mkdir(os.path.join(root, subject))
                for run in ['03', '04', '06', '14']:
                    open(os.path.join(root, subject, f'{subject}R{run}.edf'), 'w').close()
            self.assertEqual(dataset.set_records(root_path=root),
                [os.path.join(subject, f'{subject}R{run}.edf')
                 for subject in ['S001', 'S002'] for run in ['04', '06', '14']])
        with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
            dataset.set_records(root_path=root)

    def test_constants_have_matching_channel_order(self):
        self.assertEqual(len(PHYSIONET_MI_CHANNEL_LIST), 64)
        self.assertEqual(len(set(PHYSIONET_MI_CHANNEL_LIST)), 64)
        self.assertEqual([name.replace('.', '').upper()
                          for name in PHYSIONET_MI_EDF_CHANNEL_LIST],
                         PHYSIONET_MI_CHANNEL_LIST)
        coordinates = list(PHYSIONET_MI_CHANNEL_LOCATION_DICT.values())
        self.assertEqual(len({tuple(item) for item in coordinates}), 64)
        adjacency = np.asarray(PHYSIONET_MI_ADJACENCY_MATRIX)
        self.assertEqual(adjacency.shape, (64, 64))
        self.assertTrue(np.isfinite(adjacency).all())
        self.assertEqual(set(PHYSIONET_MI_CHANNEL_LOCATION_DICT),
                         set(PHYSIONET_MI_CHANNEL_LIST))

    def test_edf_reader_without_rest_annotations(self):
        rng = np.random.RandomState(42)
        data = rng.randn(64, 160 * 20) * 1e-6
        raw = mne.io.RawArray(data,
            mne.create_info(PHYSIONET_MI_EDF_CHANNEL_LIST, 160, 'eeg'),
            verbose=False)
        raw.set_annotations(mne.Annotations([1., 8.], [4., 4.], ['T1', 'T2']))
        with patch('torcheeg.datasets.module.motor_imagery.physionet_mi.mne.io.read_raw_edf',
                   return_value=raw.copy()):
            record = PhysioNetMIDataset.read_record('S001/S001R04.edf')
        self.assertEqual(record['epochs_data'].shape, (2, 64, 800))
        np.testing.assert_array_equal(record['events'], [2, 3])
        np.testing.assert_allclose(record['epochs_data'].mean(axis=1), 0, atol=1e-10)
        self.assertGreater(np.std(record['epochs_data']), 0.1)
        self.assertLess(np.std(record['epochs_data']), 10)
        rows = list(PhysioNetMIDataset.process_record(
            record='S001/S001R04.edf', **record))
        self.assertEqual([row['info']['label'] for row in rows], [0] * 4 + [1] * 4)

    def test_dataset_storage_transforms_and_parallel_workers(self):
        for mode, workers in [('pickle', 0), ('lmdb', 0), ('memory', 0),
                              ('memory', 2)]:
            with self.subTest(mode=mode, workers=workers), tempfile.TemporaryDirectory() as root:
                source = os.path.join(root, 'source')
                os.mkdir(source)
                for subject in ['S001', 'S002']:
                    folder = os.path.join(source, subject)
                    os.mkdir(folder)
                    for task in ['04', '06']:
                        open(os.path.join(folder, f'{subject}R{task}.edf'), 'w').close()
                dataset = SyntheticPhysioNetMI(
                    root_path=source, io_path=os.path.join(root, 'cache'),
                    io_mode=mode, num_worker=workers, verbose=False,
                    chunk_size=800, online_transform=transforms.ToTensor(),
                    label_transform=transforms.Select('label'))
                self.assertEqual(len(dataset), 8)
                self.assertEqual(set(dataset.info['label']), {0, 1, 2, 3})
                self.assertEqual(set(dataset.info['subject_id']), {'S001', 'S002'})
                eeg, label = dataset[7]
                self.assertEqual(tuple(eeg.shape), (64, 800))
                self.assertEqual(label, 3)
                np.testing.assert_array_equal(eeg.numpy(), np.ones((64, 800)))
                if mode == 'pickle':
                    with patch.object(SyntheticPhysioNetMI, 'read_record',
                                      side_effect=AssertionError('cache not reused')):
                        cached = SyntheticPhysioNetMI(root_path=source,
                            io_path=os.path.join(root, 'cache'), io_mode=mode,
                            verbose=False)
                    np.testing.assert_array_equal(cached[7][0], np.ones((64, 800)))

    def test_standalone_example_trains_and_holds_out_subject(self):
        example_path = Path(__file__).resolve().parents[2] / 'examples' / 'examples_physionet_mi_eegnet.py'
        spec = importlib.util.spec_from_file_location('physionet_example', example_path)
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'source'
            for subject in ['S001', 'S002']:
                folder = source / subject
                folder.mkdir(parents=True)
                for task in ['04', '06']:
                    (folder / f'{subject}R{task}.edf').touch()
            argv = ['example', '--root-path', str(source), '--cache-path',
                    str(Path(root) / 'cache'), '--epochs', '1', '--batch-size', '2']
            output = io.StringIO()
            with patch.object(example, 'PhysioNetMIDataset', SyntheticPhysioNetMI), \
                    patch.object(sys, 'argv', argv), redirect_stdout(output):
                example.main()
            result = output.getvalue()
            self.assertIn("'held_out_subject': 'S002'", result)
            self.assertIn("'train_windows': 4", result)
            self.assertIn("'test_windows': 4", result)
            self.assertIn('test_accuracy', result)

    def test_single_annotation_class_stays_t2(self):
        raw = mne.io.RawArray(np.zeros((64, 160 * 10)),
            mne.create_info(PHYSIONET_MI_EDF_CHANNEL_LIST, 160, 'eeg'),
            verbose=False)
        raw.set_annotations(mne.Annotations([1.], [4.], ['T2']))
        with patch('torcheeg.datasets.module.motor_imagery.physionet_mi.mne.io.read_raw_edf',
                   return_value=raw):
            record = PhysioNetMIDataset.read_record('S001/S001R06.edf')
        np.testing.assert_array_equal(record['events'], [3])
        rows = list(PhysioNetMIDataset.process_record(
            record='S001/S001R06.edf', **record))
        self.assertEqual([row['info']['label'] for row in rows], [3] * 4)


if __name__ == '__main__':
    unittest.main()
