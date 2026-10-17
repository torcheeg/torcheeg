import hashlib
import gc
import multiprocessing as mp
import time
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from torcheeg.datasets.module.base_dataset import BaseDataset


class RecoveryDataset(BaseDataset):
    fail_record = None

    def set_records(self, **kwargs):
        return [0, 1, 2]

    @staticmethod
    def read_record(record, trace_path, **kwargs):
        with open(trace_path, 'a') as trace:
            trace.write(f'{record}\n')
        if RecoveryDataset.fail_record == record:
            raise RuntimeError('injected record failure')
        return {}

    @staticmethod
    def process_record(record, factor=1, **kwargs):
        for index in range(2):
            key = f'{record}_{index}'
            yield {'key': key, 'eeg': np.full((1, 4), (record + index) * factor),
                   'info': {'clip_id': key, 'subject_id': record,
                            'session_id': 0, 'trial_id': 0}}


class HookFault:
    calls = 0
    fail_at = None
    block_path = None


def add_ten_with_failure(eeg):
    HookFault.calls += 1
    if HookFault.calls == 2 and HookFault.block_path is not None:
        Path(str(HookFault.block_path) + '.reached').touch()
        while Path(HookFault.block_path).exists():
            time.sleep(0.02)
    if HookFault.calls == HookFault.fail_at:
        raise RuntimeError('injected hook failure')
    return eeg + 10


class BaselineRecoveryDataset(RecoveryDataset):
    @staticmethod
    def process_record(record, **kwargs):
        baseline = f'baseline_{record}'
        yield {'key': baseline, 'eeg': np.zeros((1, 4))}
        for row in RecoveryDataset.process_record(record, **kwargs):
            row['info']['baseline_id'] = baseline
            yield row


class EmptyRecordDataset(RecoveryDataset):
    @staticmethod
    def process_record(record, **kwargs):
        if record == 0:
            return
        yield from RecoveryDataset.process_record(record, **kwargs)


class EmptyDataset(RecoveryDataset):
    def set_records(self, **kwargs):
        return []


class TransformingRecoveryDataset(RecoveryDataset):
    @staticmethod
    def process_record(record, offline_transform=None, **kwargs):
        for row in RecoveryDataset.process_record(record, **kwargs):
            if offline_transform is not None:
                row['eeg'] = offline_transform(eeg=row['eeg'])['eeg']
            yield row


def offline_factory(offset):
    return lambda eeg: {'eeg': eeg + offset}


class BlockingRecoveryDataset(RecoveryDataset):
    @staticmethod
    def read_record(record, block_path, trace_path, **kwargs):
        result = RecoveryDataset.read_record(record, trace_path, **kwargs)
        if record == 1:
            Path(str(block_path) + '.reached').touch()
            while Path(block_path).exists():
                time.sleep(0.02)
        return result


def build_in_process(root, mode, blocked=False, hook=False):
    cls = BlockingRecoveryDataset if blocked else RecoveryDataset
    kwargs = {}
    if blocked:
        kwargs['block_path'] = str(Path(root) / 'block')
    if hook:
        HookFault.calls = 0
        HookFault.block_path = str(Path(root) / 'block')
        kwargs['after_subject'] = add_ten_with_failure
    dataset = cls(io_path=str(Path(root) / 'cache'),
                  trace_path=str(Path(root) / 'trace.txt'), io_mode=mode,
                  verbose=False, **kwargs)
    assert len(dataset) == 6



class TestCacheRecovery(unittest.TestCase):
    def tearDown(self):
        RecoveryDataset.fail_record = None
        HookFault.calls = 0
        HookFault.fail_at = None
        HookFault.block_path = None

    def build(self, root, dataset_class=RecoveryDataset, mode='pickle', **kwargs):
        return dataset_class(io_path=str(Path(root) / 'cache'),
                               trace_path=str(Path(root) / 'trace.txt'),
                               io_mode=mode, verbose=False, **kwargs)

    def trace(self, root):
        return (Path(root) / 'trace.txt').read_text().splitlines()

    def fingerprints(self, path):
        return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(path).rglob('*') if p.is_file()}

    def test_retry_reuses_completed_records(self):
        with tempfile.TemporaryDirectory() as root:
            RecoveryDataset.fail_record = 1
            with self.assertRaisesRegex(RuntimeError, 'record failure'):
                self.build(root)
            RecoveryDataset.fail_record = None
            dataset = self.build(root)
            self.assertEqual(self.trace(root), ['0', '1', '1', '2'])
            self.assertEqual(len(dataset), 6)
            np.testing.assert_array_equal(dataset[5][0], np.full((1, 4), 3))

    def test_hook_retry_preserves_clean_records_without_double_application(self):
        with tempfile.TemporaryDirectory() as root:
            HookFault.fail_at = 2
            with self.assertRaisesRegex(RuntimeError, 'hook failure'):
                self.build(root, after_subject=add_ten_with_failure)
            HookFault.calls = 0
            HookFault.fail_at = None
            dataset = self.build(root, after_subject=add_ten_with_failure)
            self.assertEqual(self.trace(root), ['0', '1', '2'])
            for index in range(6):
                record, sample = divmod(index, 2)
                np.testing.assert_array_equal(dataset[index][0],
                                             np.full((1, 4), record + sample + 10))

    def test_parameter_mismatch_preserves_completed_cache(self):
        with tempfile.TemporaryDirectory() as root:
            dataset = self.build(root, factor=1)
            cache = Path(root) / 'cache'
            before = self.fingerprints(cache)
            with self.assertRaises(ValueError):
                self.build(root, factor=2)
            self.assertEqual(before, self.fingerprints(cache))
            self.assertEqual(self.trace(root), ['0', '1', '2'])
            np.testing.assert_array_equal(dataset[5][0], np.full((1, 4), 3))

    def test_reuse_all_backends_and_online_settings(self):
        for mode in ['pickle', 'lmdb', 'memmap']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                dataset = self.build(root, mode=mode)
                dataset.write_eeg('_record_0', '0_0', np.full((1, 4), 99))
                cached = self.build(root, mode=mode, num_worker=2, lazy_threshold=0,
                                    online_transform=lambda eeg: eeg,
                                    label_transform=lambda y: y)
                self.assertEqual(self.trace(root), ['0', '1', '2'])
                np.testing.assert_array_equal(cached[0][0], np.full((1, 4), 99))
                del dataset, cached
                gc.collect()

    def test_parallel_preprocessing_all_backends(self):
        for mode in ['pickle', 'lmdb', 'memmap']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                dataset = self.build(root, mode=mode, num_worker=2)
                self.assertEqual(sorted(self.trace(root)), ['0', '1', '2'])
                self.assertEqual(len(dataset), 6)
                cached = self.build(root, mode=mode)
                self.assertEqual(len(self.trace(root)), 3)
                np.testing.assert_array_equal(cached[5][0], np.full((1, 4), 3))
                del dataset, cached
                gc.collect()

    def test_partial_configuration_mismatch_preserves_resume_data(self):
        with tempfile.TemporaryDirectory() as root:
            RecoveryDataset.fail_record = 1
            with self.assertRaises(RuntimeError):
                self.build(root, factor=1)
            stage = next(Path(root).glob('.*.torcheeg-stage'))
            before = self.fingerprints(stage)
            RecoveryDataset.fail_record = None
            with self.assertRaisesRegex(ValueError, 'incomplete cache configuration'):
                self.build(root, factor=2)
            self.assertEqual(before, self.fingerprints(stage))
            dataset = self.build(root, factor=1)
            self.assertEqual(len(dataset), 6)
            self.assertEqual(self.trace(root), ['0', '1', '1', '2'])

    def test_changed_source_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'source'
            source.mkdir()
            raw = source / 'raw.txt'
            raw.write_text('original')
            dataset = self.build(root, root_path=str(source))
            cache = Path(root) / 'cache'
            before = self.fingerprints(cache)
            raw.write_text('changed source contents')
            with self.assertRaisesRegex(ValueError, 'Source metadata changed'):
                self.build(root, root_path=str(source))
            self.assertEqual(before, self.fingerprints(cache))
            self.assertEqual(len(dataset), 6)

    def test_missing_clip_baseline_or_metadata_is_not_deleted(self):
        for failure in ['clip', 'baseline', 'metadata']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                dataset = self.build(root, dataset_class=BaselineRecoveryDataset)
                record = Path(root) / 'cache' / '_record_0'
                if failure == 'clip':
                    (record / 'eeg' / '0_1').unlink()
                elif failure == 'baseline':
                    (record / 'eeg' / 'baseline_0').unlink()
                else:
                    with open(record / 'info.csv', 'a') as stream:
                        stream.write('broken,metadata\n')
                before = self.fingerprints(Path(root) / 'cache')
                with self.assertRaises(ValueError):
                    self.build(root, dataset_class=BaselineRecoveryDataset)
                self.assertEqual(before, self.fingerprints(Path(root) / 'cache'))
                del dataset

    def test_unowned_stage_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            stage = Path(root) / '.cache.torcheeg-stage'
            stage.mkdir()
            (stage / 'keep.txt').write_text('user data')
            with self.assertRaises(ValueError):
                self.build(root)
            self.assertEqual((stage / 'keep.txt').read_text(), 'user data')
            self.assertFalse((Path(root) / 'cache').exists())
            self.assertFalse((Path(root) / 'trace.txt').exists())

    def test_legacy_cache_validation_does_not_add_manifests(self):
        with tempfile.TemporaryDirectory() as root:
            dataset = self.build(root)
            cache = Path(root) / 'cache'
            for marker in cache.rglob('.torcheeg-*.json'):
                marker.unlink()
            before = self.fingerprints(cache)
            cached = self.build(root)
            self.assertEqual(before, self.fingerprints(cache))
            self.assertEqual(len(cached), len(dataset))
            self.assertEqual(self.trace(root), ['0', '1', '2'])

    def test_publication_then_reader_failure_does_not_repeat_hooks(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / 'cache'
            original_init = RecoveryDataset.init_io
            def fail_final(instance, io_path, io_size, io_mode):
                if Path(io_path) == target:
                    raise RuntimeError('injected final reader failure')
                return original_init(instance, io_path, io_size, io_mode)
            with patch.object(RecoveryDataset, 'init_io', new=fail_final):
                with self.assertRaisesRegex(RuntimeError, 'final reader failure'):
                    self.build(root, after_subject=add_ten_with_failure)
            self.assertTrue(target.is_dir())
            self.assertTrue(list(Path(root).glob('.*.torcheeg-stage')))
            self.assertEqual(HookFault.calls, 3)
            dataset = self.build(root, after_subject=add_ten_with_failure)
            self.assertEqual(HookFault.calls, 3)
            self.assertEqual(self.trace(root), ['0', '1', '2'])
            np.testing.assert_array_equal(dataset[0][0], np.full((1, 4), 10))
            self.assertFalse(list(Path(root).glob('.*.torcheeg-stage')))

    def test_unexpected_record_or_corrupt_manifest_is_preserved(self):
        for failure in ['extra', 'manifest']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                self.build(root)
                cache = Path(root) / 'cache'
                if failure == 'extra':
                    (cache / '_record_99').mkdir()
                else:
                    (cache / '.torcheeg-cache.json').write_text('{broken')
                before = self.fingerprints(cache)
                with self.assertRaises(ValueError):
                    self.build(root)
                self.assertEqual(before, self.fingerprints(cache))

    def test_empty_record_and_empty_source_selection(self):
        with tempfile.TemporaryDirectory() as root:
            dataset = self.build(root, dataset_class=EmptyRecordDataset)
            self.assertEqual(len(dataset), 4)
            np.testing.assert_array_equal(dataset[0][0], np.full((1, 4), 1))
            cached = self.build(root, dataset_class=EmptyRecordDataset)
            self.assertEqual(len(cached), 4)
            self.assertEqual(self.trace(root), ['0', '1', '2'])
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'No source records'):
                self.build(root, dataset_class=EmptyDataset)
            self.assertFalse((Path(root) / 'cache').exists())

    def test_dynamic_callback_identity_and_changed_closure(self):
        with tempfile.TemporaryDirectory() as root:
            dataset = self.build(root, dataset_class=TransformingRecoveryDataset,
                                 offline_transform=offline_factory(3))
            cached = self.build(root, dataset_class=TransformingRecoveryDataset,
                                offline_transform=offline_factory(3))
            self.assertEqual(self.trace(root), ['0', '1', '2'])
            np.testing.assert_array_equal(cached[0][0], np.full((1, 4), 3))
            before = self.fingerprints(Path(root) / 'cache')
            with self.assertRaisesRegex(ValueError, 'cache configuration'):
                self.build(root, dataset_class=TransformingRecoveryDataset,
                           offline_transform=offline_factory(4))
            self.assertEqual(before, self.fingerprints(Path(root) / 'cache'))
            self.assertEqual(len(dataset), 6)

    def test_completed_cache_can_be_read_without_source_directory(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'source'
            source.mkdir()
            (source / 'raw.txt').write_text('source')
            self.build(root, root_path=str(source))
            shutil.rmtree(source)
            dataset = self.build(root, root_path=str(source))
            self.assertEqual(len(dataset), 6)
            self.assertEqual(self.trace(root), ['0', '1', '2'])

    def test_csv_referenced_source_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'signal.dat'
            source.write_text('raw')
            csv_path = Path(root) / 'files.csv'
            csv_path.write_text(f'file_path\n{source}\n')
            dataset = self.build(root, csv_path=str(csv_path))
            source.write_text('changed raw signal')
            before = self.fingerprints(Path(root) / 'cache')
            with self.assertRaisesRegex(ValueError, 'Source metadata changed'):
                self.build(root, csv_path=str(csv_path))
            self.assertEqual(before, self.fingerprints(Path(root) / 'cache'))
            self.assertEqual(len(dataset), 6)

    def test_interrupted_cleanup_remains_resumable(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'source'
            source.mkdir()
            (source / 'raw.txt').write_text('source')
            cache = source / 'cache'
            trace = Path(root) / 'trace.txt'
            original_remove = shutil.rmtree
            def fail_retired(path, *args, **kwargs):
                if '.torcheeg-retired-' in str(path):
                    marker = next(Path(path).glob('.torcheeg-stage.json'))
                    marker.unlink()
                    raise OSError('injected partial cleanup')
                return original_remove(path, *args, **kwargs)
            with patch('torcheeg.datasets.module._cache.shutil.rmtree', new=fail_retired):
                dataset = RecoveryDataset(io_path=str(cache), trace_path=str(trace),
                    root_path=str(source), io_mode='pickle', verbose=False,
                    after_subject=add_ten_with_failure)
            self.assertTrue(list(source.glob('.*.torcheeg-retired-*')))
            self.assertEqual(HookFault.calls, 3)
            cached = RecoveryDataset(io_path=str(cache), trace_path=str(trace),
                root_path=str(source), io_mode='pickle', verbose=False,
                after_subject=add_ten_with_failure)
            self.assertFalse(list(source.glob('.*.torcheeg-retired-*')))
            self.assertEqual(HookFault.calls, 3)
            np.testing.assert_array_equal(cached[0][0], np.full((1, 4), 10))
            self.assertEqual(len(dataset), len(cached))

    def wait_for(self, path, process):
        deadline = time.monotonic() + 30
        while not path.exists() and process.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(path.exists(), f'Child did not reach checkpoint; exit={process.exitcode}')

    def stop(self, process):
        if process.is_alive():
            process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(10)
        self.assertFalse(process.is_alive())

    def test_abrupt_record_termination_resumes_all_backends(self):
        for mode in ['pickle', 'lmdb', 'memmap']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                block = Path(root) / 'block'
                block.touch()
                process = mp.get_context('spawn').Process(
                    target=build_in_process, args=(root, mode, True))
                process.start()
                try:
                    self.wait_for(Path(str(block) + '.reached'), process)
                    self.assertFalse((Path(root) / 'cache').exists())
                    self.stop(process)
                    block.unlink()
                    dataset = self.build(root, mode=mode,
                        dataset_class=BlockingRecoveryDataset, block_path=str(block))
                    self.assertEqual(self.trace(root), ['0', '1', '1', '2'])
                    np.testing.assert_array_equal(dataset[5][0], np.full((1, 4), 3))
                    del dataset
                    gc.collect()
                finally:
                    self.stop(process)

    def test_abrupt_hook_termination_never_reapplies_to_dirty_data(self):
        for mode in ['pickle', 'lmdb', 'memmap']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                block = Path(root) / 'block'
                block.touch()
                process = mp.get_context('spawn').Process(
                    target=build_in_process, args=(root, mode, False, True))
                process.start()
                try:
                    self.wait_for(Path(str(block) + '.reached'), process)
                    self.assertFalse((Path(root) / 'cache').exists())
                    self.stop(process)
                    block.unlink()
                    HookFault.calls = 0
                    HookFault.block_path = None
                    dataset = self.build(root, mode=mode, after_subject=add_ten_with_failure)
                    self.assertEqual(self.trace(root), ['0', '1', '2'])
                    for index in range(6):
                        record, sample = divmod(index, 2)
                        np.testing.assert_array_equal(dataset[index][0],
                            np.full((1, 4), record + sample + 10))
                    del dataset
                    gc.collect()
                finally:
                    self.stop(process)

    def test_concurrent_processes_publish_once(self):
        for method in ['spawn', 'fork']:
            if method not in mp.get_all_start_methods():
                continue
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                context = mp.get_context(method)
                processes = [context.Process(target=build_in_process,
                             args=(root, 'pickle')) for _ in range(2)]
                for process in processes:
                    process.start()
                try:
                    for process in processes:
                        process.join(30)
                        self.assertEqual(process.exitcode, 0)
                    self.assertEqual(self.trace(root), ['0', '1', '2'])
                    self.assertEqual(len(self.build(root)), 6)
                finally:
                    for process in processes:
                        self.stop(process)


if __name__ == '__main__':
    unittest.main()
