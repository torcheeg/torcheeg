import json
import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from filelock import FileLock

from torcheeg import transforms
from torcheeg.datasets import BaseDataset, DEAPDataset


class _CachedDataset(BaseDataset):

    def set_records(self, root_path, **kwargs):
        return sorted(str(path) for path in Path(root_path).glob('*.npy'))

    @staticmethod
    def read_record(record, **kwargs):
        return {'samples': np.load(record)}

    @staticmethod
    def process_record(record,
                       samples,
                       scale=1,
                       offline_transform=None,
                       fail=False,
                       **kwargs):
        if fail:
            raise RuntimeError('Interrupted writer')
        for index, sample in enumerate(samples):
            eeg = sample * scale
            if offline_transform is not None:
                eeg = offline_transform(eeg=eeg)['eeg']
            yield {
                'eeg': eeg,
                'key': str(index),
                'info': {
                    'clip_id': str(index),
                    'subject_id': 0,
                    'trial_id': 0
                }
            }


def _dataset_worker(source, path, queue):
    try:
        dataset = _CachedDataset(root_path=source,
                                 io_path=path,
                                 io_mode='pickle',
                                 verbose=False)
        queue.put((len(dataset), float(dataset[0][0].sum())))
    except Exception as error:
        queue.put(repr(error))


def _add_one(samples):
    return samples + 1


def _fail_postprocessing(samples):
    raise RuntimeError('Interrupted postprocessing')


class TestDatasetCache(unittest.TestCase):

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._root = Path(self._directory.name)
        self._source = self._root / 'source'
        self._source.mkdir()
        self._samples = np.arange(24).reshape(3, 2, 4)
        np.save(self._source / 'record.npy', self._samples)
        self._path = self._root / 'cache'

    def _dataset(self, **kwargs):
        parameters = dict(root_path=str(self._source),
                          io_path=str(self._path),
                          io_mode='pickle',
                          verbose=False)
        parameters.update(kwargs)
        return _CachedDataset(**parameters)

    def test_disk_modes_preserve_samples_and_reuse(self):
        for mode in ('pickle', 'lmdb'):
            with self.subTest(mode=mode):
                path = self._root / mode
                first = self._dataset(io_path=path,
                                      io_mode=mode,
                                      lazy_threshold=0,
                                      after_trial=_add_one)
                np.testing.assert_array_equal(first[0][0], self._samples[0] + 1)
                manifest = path / 'manifest.json'
                before = manifest.read_bytes()
                repeated = self._dataset(
                    io_path=path,
                    io_mode=mode,
                    lazy_threshold=0,
                    after_trial=_add_one,
                    timeout=0,
                    num_worker=1,
                    online_transform=transforms.ToTensor(),
                    label_transform=transforms.Select('subject_id'))
                np.testing.assert_array_equal(repeated[0][0], self._samples[0] + 1)
                self.assertEqual(before, manifest.read_bytes())
                self.assertEqual(set(json.loads(before)), {'fingerprint', 'files'})

    def test_offline_parameters_and_hooks_invalidate_cache(self):
        self._dataset()
        for kwargs in ({
                'scale': 2
        }, {
                'offline_transform': transforms.MeanStdNormalize()
        }, {
                'after_trial': _add_one
        }):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                    ValueError, 'changed'):
                self._dataset(**kwargs)

    def test_lambda_transform_configuration_is_stable(self):

        def make_transform(offset):
            return lambda eeg: {'eeg': eeg + offset}

        first = self._dataset(offline_transform=make_transform(1))
        repeated = self._dataset(offline_transform=make_transform(1))
        np.testing.assert_array_equal(first[0][0], repeated[0][0])
        with self.assertRaisesRegex(ValueError, 'changed'):
            self._dataset(offline_transform=make_transform(2))

    def test_source_content_is_checked_even_if_size_and_time_match(self):
        self._dataset()
        file = self._source / 'record.npy'
        stat = file.stat()
        np.save(file, self._samples + 1)
        os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'changed'):
            self._dataset()

    def test_changed_or_missing_outputs_are_rejected(self):
        self._dataset()
        sample = next((self._path / '_record_0' / 'eeg').iterdir())
        original = sample.read_bytes()
        sample.write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'Corrupt'):
            self._dataset()
        sample.write_bytes(original)
        sample.unlink()
        with self.assertRaisesRegex(ValueError, 'changed'):
            self._dataset()

    def test_failed_writer_and_postprocessing_can_retry(self):
        for kwargs, message in (({
                'fail': True
        }, 'Interrupted writer'), ({
                'after_trial': _fail_postprocessing
        }, 'Interrupted postprocessing')):
            with self.subTest(kwargs=kwargs):
                path = self._root / message.replace(' ', '_')
                with self.assertRaisesRegex(RuntimeError, message):
                    self._dataset(io_path=path, **kwargs)
                self.assertFalse(path.exists())
                self.assertFalse(list(self._root.glob('.torcheeg-cache-*')))
                recovered = self._dataset(io_path=path)
                np.testing.assert_array_equal(recovered[0][0], self._samples[0])

    def test_unknown_existing_cache_is_preserved(self):
        self._path.mkdir()
        file = self._path / 'old.csv'
        file.write_text('old data')
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self._dataset()
        self.assertEqual(file.read_text(), 'old data')

    def test_timeout_is_public_and_validated(self):
        with FileLock(str(self._path) + '.lock'):
            with self.assertRaisesRegex(RuntimeError, 'Timeout'):
                self._dataset(timeout=0)
        for value, error in ((True, TypeError), ('1', TypeError), (-2, ValueError)):
            with self.subTest(value=value), self.assertRaises(error):
                self._dataset(timeout=value)
        # Concrete datasets expose the same option before attempting source IO.
        with self.assertRaises(TypeError):
            DEAPDataset(timeout='1', verbose=False)

    def test_competing_processes_publish_one_complete_cache(self):
        context = multiprocessing.get_context('spawn')
        queue = context.Queue()
        processes = [
            context.Process(target=_dataset_worker,
                            args=(str(self._source), str(self._path), queue))
            for _ in range(2)
        ]
        try:
            for process in processes:
                process.start()
            results = [queue.get(timeout=60) for _ in processes]
            self.assertEqual(results, [(3, float(self._samples[0].sum()))] * 2)
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join()
            queue.close()
            queue.join_thread()

    def test_memory_mode_retains_signals_without_persistent_manifest(self):
        for workers in (0, 2):
            with self.subTest(workers=workers):
                dataset = self._dataset(io_mode='memory',
                                        num_worker=workers,
                                        after_trial=_add_one)
                np.testing.assert_array_equal(dataset[1][0], self._samples[1] + 1)
                self.assertFalse(self._path.exists())

    def test_omitted_paths_create_separate_caches(self):
        first = self._dataset(io_path=None)
        second = self._dataset(io_path=None)
        self.assertNotEqual(first.io_path, second.io_path)
        # BaseDataset defaults are relative to the working directory.
        import shutil
        self.addCleanup(shutil.rmtree, first.io_path)
        self.addCleanup(shutil.rmtree, second.io_path)
        self.assertTrue(Path(first.io_path, 'manifest.json').is_file())
        self.assertTrue(Path(second.io_path, 'manifest.json').is_file())
        self.addCleanup(Path(first.io_path + '.lock').unlink, missing_ok=True)
        self.addCleanup(Path(second.io_path + '.lock').unlink, missing_ok=True)


if __name__ == '__main__':
    unittest.main()
