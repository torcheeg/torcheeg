import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from torcheeg.datasets import BaseData, IODataset
from torcheeg.io import (LMDBEEGIO, CSVInfoIO, JSONLInfoIO, MemoryEEGIO,
                         MemoryInfoIO, NumpyEEGIO, SQLiteInfoIO)
from torcheeg.io.utils import ensure_cache
from torcheeg.runners import SequentialRunner


class TestIOCache(unittest.TestCase):

    def test_persistent_pairs_reopen_and_reuse(self):
        for eeg_class in (LMDBEEGIO, NumpyEEGIO):
            for info_class in (CSVInfoIO, JSONLInfoIO, SQLiteInfoIO):
                with self.subTest(eeg=eeg_class, info=info_class), tempfile.TemporaryDirectory() as root:
                    eeg_path, info_path = (
                        str(Path(root, 'eeg')), str(Path(root, 'info')))
                    eeg, info = (eeg_class(eeg_path), info_class(info_path))
                    _build_cache(_WindowData(), eeg, info)
                    first = IODataset(eeg, info)
                    manifest = Path(root, 'manifest.json')
                    before = manifest.read_bytes()
                    self.assertEqual(set(json.loads(before)),
                                     {'fingerprint', 'files'})
                    eeg.close()
                    eeg, info = (eeg_class(eeg_path), info_class(info_path))
                    _build_cache(_WindowData(), eeg, info)
                    repeated = IODataset(eeg, info)
                    self.assertEqual(len(repeated), 4)
                    np.testing.assert_array_equal(
                        first[0]['eeg'], repeated[0]['eeg'])
                    self.assertEqual(before, manifest.read_bytes())
                    eeg.close()

    def test_failed_creation_can_retry_without_reusing_partial_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            eeg = NumpyEEGIO(str(Path(root, 'eeg')))
            info = CSVInfoIO(str(Path(root, 'info.csv')))
            with self.assertRaisesRegex(RuntimeError, 'record failed'):
                _build_cache(_WindowData(fail=True), eeg, info)
            self.assertEqual(len(eeg), 0)
            self.assertEqual(len(info), 0)
            self.assertFalse(list(Path(root).glob('.torcheeg-cache-*')))
            _build_cache(_WindowData(), eeg, info)
            self.assertEqual(len(info), 4)

    def test_changed_configuration_and_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            eeg = NumpyEEGIO(str(Path(root, 'eeg')))
            info = CSVInfoIO(str(Path(root, 'info.csv')))
            _build_cache(_WindowData(), eeg, info)
            with self.assertRaisesRegex(ValueError, 'changed'):
                _build_cache(_WindowData(num_records=3), eeg, info)
            next(Path(eeg.io_path).glob('*.npy')).write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'Corrupt'):
                _build_cache(_WindowData(), eeg, info)

    def test_memory_and_persistent_io_can_be_composed(self):
        with tempfile.TemporaryDirectory() as root:
            for eeg, info in ((MemoryEEGIO(), CSVInfoIO(str(Path(root, 'info.csv')))), (NumpyEEGIO(str(Path(root, 'eeg'))), MemoryInfoIO())):
                _build_cache(_WindowData(), eeg, info)
                self.assertEqual(len(IODataset(eeg, info)), 4)

    def test_omitted_io_paths_are_unique_and_can_be_reopened(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as root, patch('torcheeg.io.eeg_io.numpy.get_random_dir_path', side_effect=[str(Path(root, 'eeg_1')), str(Path(root, 'eeg_2'))]), patch('torcheeg.io.info_io.base.get_random_dir_path', side_effect=[str(Path(root, 'info_1')), str(Path(root, 'info_2'))]):
            pairs = [(NumpyEEGIO(), CSVInfoIO()) for _ in range(2)]
            self.assertNotEqual(pairs[0][0].io_path, pairs[1][0].io_path)
            self.assertNotEqual(pairs[0][1].io_path, pairs[1][1].io_path)
            for eeg, info in pairs:
                _build_cache(_WindowData(), eeg, info)
                reopened_eeg = NumpyEEGIO(eeg.io_path)
                reopened_info = CSVInfoIO(info.io_path)
                _build_cache(_WindowData(), reopened_eeg, reopened_info)
                self.assertEqual(
                    len(IODataset(reopened_eeg, reopened_info)), 4)

    def test_separate_storage_directories(self):
        with tempfile.TemporaryDirectory() as root:
            eeg = NumpyEEGIO(str(Path(root, 'signals', 'eeg')))
            info = JSONLInfoIO(str(Path(root, 'metadata', 'info.jsonl')))
            _build_cache(_WindowData(), eeg, info)
            _build_cache(_WindowData(), eeg, info)
            self.assertEqual(len(IODataset(eeg, info)), 4)

    def test_failed_publication_restores_both_original_io_objects(self):
        import os
        from unittest.mock import patch
        for in_memory in (False, True):
            with self.subTest(in_memory=in_memory), tempfile.TemporaryDirectory() as root:
                eeg = MemoryEEGIO() if in_memory else NumpyEEGIO(str(Path(root, 'eeg')))
                info = CSVInfoIO(str(Path(root, 'info.csv')))
                replace = os.replace

                def fail_metadata(source, destination):
                    if str(source).endswith('/data') and destination == info.io_path:
                        raise OSError('publication failed')
                    return replace(source, destination)
                with patch('torcheeg.io.utils.os.replace', side_effect=fail_metadata):
                    with self.assertRaisesRegex(OSError, 'publication failed'):
                        _build_cache(_WindowData(), eeg, info)
                self.assertEqual(len(eeg), 0)
                self.assertEqual(len(info), 0)
                self.assertFalse(list(Path(root).glob('.torcheeg-cache-*')))
                _build_cache(_WindowData(), eeg, info)
                self.assertEqual(len(info), 4)


class _WindowData(BaseData):
    def set_records(self, num_records=2, **kwargs):
        return list(range(num_records))

    def read_record(self, record, **kwargs):
        return {}

    def process_record(self, record, fail=False, **kwargs):
        if fail:
            raise RuntimeError('record failed')
        for index in range(2):
            yield {'eeg': np.full((2, 4), index), 'info': {'subject_id': str(record)}}


def _build_cache(data, eeg, info):
    def build(target_eeg, target_info):
        IODataset(target_eeg, target_info).extend(SequentialRunner()(data))
    ensure_cache(eeg, info, json.dumps(
        data.configuration, sort_keys=True), build)
