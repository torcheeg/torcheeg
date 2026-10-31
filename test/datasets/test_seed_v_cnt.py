"""Exercise the actual Neuroscan reader with known int16/int32 wire values."""

import inspect
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import warnings

import mne
import numpy as np
from packaging.version import Version

from torcheeg.datasets import SEEDVDataset


def write_cnt(path, data_format='int16', sentinel=False):
    channels, samples = 66, 20
    values = np.arange(channels * samples).reshape(samples, channels) - 600
    dtype = np.dtype('<i2' if data_format == 'int16' else '<i4')
    event_position = 900 + 75 * channels + values.size * dtype.itemsize
    data = bytearray(event_position + 9)
    struct.pack_into('<H', data, 370, channels)
    struct.pack_into('<H', data, 376, 250)
    struct.pack_into('<I', data, 864, 4294966564 if sentinel else samples)
    struct.pack_into('<i', data, 886, event_position)
    struct.pack_into('<i', data, 894, 1)
    names = ['EEG%02d' % i for i in range(62)] + ['M1', 'M2', 'VEO', 'HEO']
    for i, name in enumerate(names):
        start = 900 + 75 * i
        data[start : start + len(name)] = name.encode('ascii')
        angle = 2 * np.pi * i / channels
        struct.pack_into('<ff', data, start + 19, np.cos(angle), np.sin(angle))
        struct.pack_into('<f', data, start + 59, 204.8)
        struct.pack_into('<f', data, start + 71, 1.0)
    payload = values.astype(dtype).tobytes()
    data[900 + 75 * channels : event_position] = payload
    struct.pack_into('<Bii', data, event_position, 1, 0, 0)
    path.write_bytes(data)
    # MNE calibrates with the exact float32 sensitivity stored in the header.
    return values[:, :62].T * (float(np.float32(204.8)) * 1e-6 / 204.8)


class TestSEEDVCNT(unittest.TestCase):
    def test_actual_reader_auto_and_both_explicit_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            for data_format in ('int16', 'int32'):
                path = Path(directory) / (data_format + '.cnt')
                expected = write_cnt(path, data_format)
                for option in ('auto', data_format):
                    with self.subTest(data_format=data_format, option=option):
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', RuntimeWarning)
                            actual = SEEDVDataset.read_record(
                                str(path), data_format=option
                            )['data_matrix']
                        np.testing.assert_allclose(
                            actual, expected, rtol=1e-14, atol=1e-15
                        )

    def test_invalid_format_preserves_existing_cache_and_never_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'cache'
            cache.mkdir()
            marker = cache / 'existing-data'
            marker.write_bytes(b'preserve this cache')
            with patch('mne.io.read_raw_cnt') as reader:
                for value in ('float32', '', None, 16):
                    with self.assertRaises(ValueError):
                        SEEDVDataset(data_format=value, io_path=str(cache))
                    with self.assertRaises(ValueError):
                        SEEDVDataset.read_record('missing.cnt', data_format=value)
                reader.assert_not_called()
            self.assertEqual(marker.read_bytes(), b'preserve this cache')
            self.assertEqual(list(cache.iterdir()), [marker])

    def test_format_is_keyword_only_and_part_of_cache_identity(self):
        from torcheeg.datasets.module.base_dataset import BaseDataset
        from torcheeg.datasets.module._cache import configuration_signature

        parameter = inspect.signature(SEEDVDataset).parameters['data_format']
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        signatures = []
        with patch.object(BaseDataset, '__init__', return_value=None) as initialize:
            for value in ('auto', 'int16', 'int32'):
                dataset = SEEDVDataset(data_format=value, io_path='identity-only-cache')
                parameters = initialize.call_args.kwargs
                self.assertEqual(parameters['data_format'], value)
                self.assertEqual(dataset.repr_body['data_format'], value)
                signatures.append(configuration_signature(dataset, parameters))
        self.assertEqual(len(set(signatures)), 3)

    def test_oversized_header_is_rejected_before_dense_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.cnt'
            write_cnt(path)
            raw = MagicMock()
            raw.__enter__.return_value = raw
            raw.ch_names = ['EEG'] * 66
            raw.n_times = 4294966564
            with patch('mne.io.read_raw_cnt', return_value=raw):
                with self.assertRaisesRegex(ValueError, 'file-size bound'):
                    SEEDVDataset.read_record(str(path), data_format='int16')
            raw.get_data.assert_not_called()
            raw.drop_channels.assert_not_called()
            raw.__exit__.assert_called_once()

    def test_exact_numpy2_overflow_has_actionable_cause_without_retry(self):
        error = RuntimeError(
            'OverflowError: Python integer 283467793224 out of bounds for int32'
        )
        with patch('mne.io.read_raw_cnt', side_effect=error) as reader:
            with self.assertRaisesRegex(ValueError, 'data_format') as raised:
                SEEDVDataset.read_record('header.cnt')
            self.assertIs(raised.exception.__cause__, error)
            reader.assert_called_once_with('header.cnt', data_format='auto')
        unrelated = RuntimeError('unrelated CNT reader error')
        with patch('mne.io.read_raw_cnt', side_effect=unrelated):
            with self.assertRaises(RuntimeError) as raised:
                SEEDVDataset.read_record('header.cnt')
            self.assertIs(raised.exception, unrelated)

    def test_old_reader_is_not_constructed_for_impossible_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sentinel.cnt'
            write_cnt(path, sentinel=True)
            with patch.object(mne, '__version__', '1.9.0'):
                with patch('mne.io.read_raw_cnt') as reader:
                    with self.assertRaisesRegex(ValueError, 'file-size bound'):
                        SEEDVDataset.read_record(str(path), data_format='int16')
                    reader.assert_not_called()

    def test_real_sentinel_header_never_allocates_unsigned_sample_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sentinel.cnt'
            expected = write_cnt(path, sentinel=True)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                if Version(mne.__version__) == Version('1.9.0') and Version(
                    np.__version__
                ) >= Version('2'):
                    with self.assertRaisesRegex(ValueError, 'Cannot infer') as raised:
                        SEEDVDataset.read_record(str(path))
                    self.assertIn('283467793224', str(raised.exception.__cause__))
                if Version(mne.__version__) >= Version('1.12'):
                    actual = SEEDVDataset.read_record(str(path), data_format='int16')[
                        'data_matrix'
                    ]
                    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
                else:
                    with self.assertRaisesRegex(ValueError, 'file-size bound'):
                        SEEDVDataset.read_record(str(path), data_format='int16')
