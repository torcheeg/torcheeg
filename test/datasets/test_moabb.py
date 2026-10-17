import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from moabb.datasets.fake import FakeDataset
from moabb.paradigms import LeftRightImagery

from torcheeg import transforms
from torcheeg.datasets.moabb import MOABBDataset


def make_source():
    return FakeDataset(event_list=['left_hand', 'right_hand'], n_sessions=2,
                       n_runs=1, n_subjects=2, seed=42, sfreq=64,
                       duration=20, n_events=4)


def double_raw(raw):
    raw.apply_function(lambda data: data * 2, picks='eeg')
    return raw


class TestMOABB(unittest.TestCase):
    def test_public_paradigm_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            source = make_source()
            paradigm = LeftRightImagery(fmin=4, fmax=24, tmin=0, tmax=1,
                                        resample=64)
            expected, labels, metadata = paradigm.get_data(source, subjects=[1])
            source.subject_list = [1]
            arguments = dict(dataset=source, paradigm=paradigm,
                             io_path=str(Path(folder) / 'cache'),
                             io_mode='pickle', verbose=False)
            dataset = MOABBDataset(**arguments)
            self.assertEqual(len(dataset), len(expected))
            for index, signal in enumerate(expected):
                np.testing.assert_allclose(dataset[index][0], signal, rtol=1e-12, atol=1e-12)
            self.assertEqual(dataset.info.label.tolist(), labels.tolist())
            self.assertEqual(dataset.info.session_id.tolist(), metadata.session.tolist())
            with patch.object(type(paradigm), 'get_data', side_effect=AssertionError('reread')) as read:
                cached = MOABBDataset(**arguments)
                self.assertEqual(len(cached), len(dataset))
                read.assert_not_called()

    def test_epoch_windows_keep_distinct_trial_ids(self):
        signals = np.arange(2 * 3 * 12).reshape(2, 3, 12)
        metadata = pd.DataFrame({'subject': [1, 1], 'session': ['a', 'a'],
                                 'run': ['0', '0']})
        samples = list(MOABBDataset.process_record(record=1, samples=signals,
            labels=np.array(['left', 'right']), metadata=metadata,
            chunk_size=6, overlap=3))
        self.assertEqual(len(samples), 6)
        self.assertEqual(len({item['key'] for item in samples}), 6)
        for trial in [0, 1]:
            group = samples[trial * 3:(trial + 1) * 3]
            self.assertEqual(len({item['info']['trial_id'] for item in group}), 1)
            for window, start in zip(group, [0, 3, 6]):
                np.testing.assert_array_equal(window['eeg'], signals[trial, :, start:start + 6])
                self.assertEqual(window['info']['start_at'], start)
        self.assertNotEqual(samples[0]['info']['trial_id'], samples[3]['info']['trial_id'])

    def test_two_subjects_storage_modes_parallel_and_transforms(self):
        for mode in ['pickle', 'lmdb', 'memmap', 'memory']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                dataset = MOABBDataset(dataset=make_source(),
                    paradigm=LeftRightImagery(fmin=4, fmax=24, tmin=0, tmax=1,
                                              resample=64),
                    chunk_size=32, overlap=16, io_mode=mode,
                    io_path=str(Path(folder) / 'cache'), num_worker=2,
                    online_transform=transforms.ToTensor(),
                    label_transform=transforms.Select('label'), verbose=False)
                self.assertEqual(set(dataset.info.subject_id), {1, 2})
                self.assertEqual(set(dataset.info.session_id), {'0', '1'})
                signal, label = dataset[0]
                self.assertEqual(tuple(signal.shape), (3, 32))
                self.assertIn(label, ['left_hand', 'right_hand'])
                for _, group in dataset.info.groupby(['subject_id', 'trial_id']):
                    self.assertEqual(len(set(group.session_id)), 1)
                    self.assertEqual(len(set(group.label)), 1)

    def test_raw_hook_precedes_moabb_preprocessing(self):
        source = make_source()
        source.subject_list = [1]
        paradigm = LeftRightImagery(fmin=4, fmax=24, tmin=0, tmax=1, resample=64)
        original, _, _ = paradigm.get_data(source, subjects=[1])
        with tempfile.TemporaryDirectory() as folder:
            dataset = MOABBDataset(dataset=source, paradigm=paradigm,
                before_trial=double_raw, io_mode='memory', verbose=False,
                io_path=str(Path(folder) / 'unused'))
            np.testing.assert_allclose(dataset[0][0], original[0] * 2,
                                       rtol=1e-10, atol=1e-10)
        unchanged, _, _ = paradigm.get_data(source, subjects=[1])
        np.testing.assert_array_equal(unchanged, original)

    def test_invalid_windows_fail_before_source_access(self):
        for chunk, overlap in [(0, 0), (-2, 0), (8, 8), (8, -1), (8, 1.5), (True, 0)]:
            with self.subTest(chunk=chunk, overlap=overlap), tempfile.TemporaryDirectory() as folder:
                source = make_source()
                with patch.object(source, 'get_data', side_effect=AssertionError('source access')):
                    with self.assertRaises(ValueError):
                        MOABBDataset(dataset=source, paradigm=LeftRightImagery(),
                            io_mode='memory', io_path=str(Path(folder) / 'cache'),
                            chunk_size=chunk, overlap=overlap, verbose=False)

    def test_download_directory_must_be_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / 'cache'
            for download in [cache, cache / 'raw']:
                with self.subTest(download=download), self.assertRaises(ValueError):
                    MOABBDataset(dataset=make_source(), paradigm=LeftRightImagery(),
                        io_path=str(cache), download_path=str(download), verbose=False)
            self.assertFalse(cache.exists())

    def test_explicit_download_and_source_change_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            source = make_source()
            source.subject_list = [1]
            raw = Path(folder) / 'raw'
            raw.mkdir()
            recording = raw / 'recording.txt'
            recording.write_text('original source')
            arguments = dict(dataset=source,
                paradigm=LeftRightImagery(fmin=4, fmax=24, tmin=0, tmax=1, resample=64),
                io_path=str(Path(folder) / 'cache'), download_path=str(raw),
                io_mode='pickle', verbose=False)
            with patch.object(type(source), 'download', autospec=True) as download:
                dataset = MOABBDataset(**arguments)
                download.assert_called_once_with(source, subject_list=[1], path=str(raw),
                                                  update_path=True, verbose=False)
                cached = MOABBDataset(**arguments)
                self.assertEqual(len(dataset), len(cached))
                self.assertEqual(download.call_count, 1)
                recording.write_text('changed source with a new size')
                with self.assertRaisesRegex(ValueError, 'Source metadata changed'):
                    MOABBDataset(**arguments)
                self.assertEqual(recording.read_text(), 'changed source with a new size')

    def test_filter_bank_windows_preserve_filter_axis(self):
        signals = np.arange(2 * 3 * 12 * 2).reshape(2, 3, 12, 2)
        metadata = pd.DataFrame({'subject': [1, 1], 'session': ['0', '1'],
                                 'run': ['0', '0']})
        items = list(MOABBDataset.process_record(record=1, samples=signals,
            labels=np.array(['left', 'right']), metadata=metadata,
            chunk_size=6, overlap=0))
        self.assertEqual(len(items), 4)
        np.testing.assert_array_equal(items[1]['eeg'], signals[0, :, 6:12, :])
        self.assertEqual(items[2]['info']['session_id'], '1')

    def test_invalid_upstream_output_is_rejected(self):
        from unittest.mock import Mock
        source = make_source()
        valid = pd.DataFrame({'subject': [1], 'session': ['0'], 'run': ['0']})
        cases = [
            (np.zeros((1, 3, 8)), ['left', 'right'], valid),
            (np.zeros((1, 3, 8)), ['left'], valid.drop(columns='run')),
            (np.zeros((1, 3, 8)), ['left'], valid.assign(subject=2)),
            (np.zeros((1, 3, 8)), ['left'], valid.assign(run=None)),
            (np.zeros((3, 8)), ['left'], valid),
        ]
        for result in cases:
            with self.subTest(shape=result[0].shape), self.assertRaises(ValueError):
                MOABBDataset.read_record(record=1, dataset=source,
                                         paradigm=Mock(get_data=Mock(return_value=result)))

    def test_existing_raw_files_survive_processing_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            source = make_source()
            source.subject_list = [1]
            raw = Path(folder) / 'raw'
            raw.mkdir()
            recording = raw / 'recording.txt'
            recording.write_text('preserve this recording')
            def fail(eeg):
                raise RuntimeError('injected offline failure')
            with patch.object(type(source), 'download', autospec=True):
                with self.assertRaisesRegex(RuntimeError, 'offline failure'):
                    MOABBDataset(dataset=source,
                        paradigm=LeftRightImagery(fmin=4, fmax=24, tmin=0, tmax=1),
                        io_path=str(Path(folder) / 'cache'), download_path=str(raw),
                        io_mode='pickle', offline_transform=fail, verbose=False)
            self.assertEqual(recording.read_text(), 'preserve this recording')
            self.assertFalse((Path(folder) / 'cache').exists())

    def test_legacy_run_ids_remain_readable(self):
        with tempfile.TemporaryDirectory() as folder:
            source = make_source()
            source.subject_list = [1]
            arguments = dict(dataset=source,
                paradigm=LeftRightImagery(fmin=4, fmax=24, tmin=0, tmax=1),
                io_path=str(Path(folder) / 'cache'), io_mode='pickle', verbose=False)
            original = MOABBDataset(**arguments)
            cache = Path(arguments['io_path'])
            for manifest in cache.rglob('.torcheeg-*.json'):
                manifest.unlink()
            for metadata in cache.glob('_record_*/info.csv'):
                frame = pd.read_csv(metadata)
                frame['trial_id'] = frame['run_id']
                frame.drop(columns='run_id').to_csv(metadata, index=False)
            with self.assertLogs('torcheeg', level='WARNING'):
                legacy = MOABBDataset(**arguments)
            self.assertEqual(len(legacy), len(original))
            self.assertEqual(legacy.info.run_id.tolist(), legacy.info.trial_id.map(str).tolist())


if __name__ == '__main__':
    unittest.main()
