import os
import tempfile
import unittest

import numpy as np
import torch

from torcheeg.io import LMDBEEGIO, BaseEEGIO, MemoryEEGIO, PickleEEGIO


def _assert_backend(test, backend):
    with tempfile.TemporaryDirectory() as directory:
        io = backend(directory)
        test.addCleanup(io.close)
        signal = np.arange(12, dtype=np.float32).reshape(3, 4)
        test.assertEqual(len(io), 0)
        test.assertEqual(io.write_eeg(signal, '001'), '001')
        np.testing.assert_array_equal(io.read_eeg('001'), signal)
        stored_key = io.write_eeg(signal + 1, '001', record='a')
        np.testing.assert_array_equal(io.read_eeg(stored_key), signal + 1)
        io.write_eeg(signal + 2, '001', record='b')
        np.testing.assert_array_equal(
            io.read_eeg('001', record='a'), signal + 1)
        np.testing.assert_array_equal(
            io.read_eeg('001', record='b'), signal + 2)
        io.write_eeg(signal + 3, '001')
        test.assertEqual(len(io), 3)
        test.assertEqual(len(io.keys()), 3)
        test.assertEqual(len(io.eegs()), 3)
        np.testing.assert_array_equal(io.read_eeg('001'), signal + 3)
        with test.assertRaises(KeyError):
            io.read_eeg('missing')
        with test.assertRaises(TypeError):
            io.write_eeg(signal, 1)
        with test.assertRaises(ValueError):
            io.write_eeg(signal, '')
        with test.assertRaises(ValueError):
            io.write_eeg(None, 'none')
        clone = io.clone(directory + '/clone')
        test.addCleanup(clone.close)
        test.assertEqual(len(clone), 0)


class TestBaseEEGIO(unittest.TestCase):
    def test_required_methods_identify_the_class(self):
        io = BaseEEGIO()
        for method, args in (('keys', ()), ('read_eeg', ('x',)),
                             ('write_eeg', (0, 'x'))):
            with self.subTest(method=method), self.assertRaisesRegex(
                    NotImplementedError, 'Method ' + method + ' is not implemented in class BaseEEGIO'):
                getattr(io, method)(*args)
        io.close()

    def test_record_namespace_does_not_alias_local_keys(self):
        self.assertNotEqual(BaseEEGIO.encode_key('b/c', 'a'),
                            BaseEEGIO.encode_key('c', 'a/b'))


def _make_backend(io_path=None, io_size=1048576, io_mode='lmdb'):
    if io_mode == 'lmdb':
        return LMDBEEGIO(io_path, io_size)
    return {'memory': MemoryEEGIO, 'pickle': PickleEEGIO}[io_mode](io_path)


class TestInputValidation(unittest.TestCase):

    def test_io_backends_use_the_same_input_exceptions(self):
        for mode in ('memory', 'pickle', 'lmdb'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as path:
                io = _make_backend(path, io_mode=mode)
                try:
                    with self.assertRaises(ValueError):
                        io.write_eeg(None, key='invalid')
                    with self.assertRaises(KeyError):
                        io.read_eeg('missing')
                    key = io.write_eeg(np.zeros((2, 3)), key='sample')
                    np.testing.assert_array_equal(
                        io.read_eeg(key), np.zeros((2, 3)))
                finally:
                    del io


class TestEEGIOBackends(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._io_path = self._directory.name

    def test_init(self):
        io_io_path = os.path.join(self._io_path, 'signals')
        io_mode_space = ['pickle', 'lmdb']
        for io_mode in io_mode_space:
            io = _make_backend(io_path=os.path.join(
                io_io_path, io_mode), io_mode=io_mode)
            self.assertEqual(len(io), 0)
            self.assertTrue(os.path.exists(io_io_path))
            del io
            io = _make_backend(io_path=os.path.join(
                io_io_path, io_mode), io_mode=io_mode)
            self.assertEqual(len(io), 0)

    def test_write_eeg(self):
        io_io_path = os.path.join(self._io_path, 'signals')
        io_mode_space = ['pickle', 'lmdb']
        for io_mode in io_mode_space:
            io = _make_backend(io_path=os.path.join(
                io_io_path, io_mode), io_mode=io_mode)
            eeg = np.random.randn(32, 128)
            self.assertEqual(io.write_eeg(eeg, key='0'), '0')
            np.testing.assert_array_equal(io.read_eeg('0'), eeg)
            eeg = np.random.randn(32, 128)
            self.assertEqual(io.write_eeg(eeg, key='1'), '1')
            np.testing.assert_array_equal(io.read_eeg('1'), eeg)

    def test_len(self):
        io_io_path = os.path.join(self._io_path, 'signals')
        io_mode_space = ['pickle', 'lmdb']
        for io_mode in io_mode_space:
            io = _make_backend(io_path=os.path.join(
                io_io_path, io_mode), io_mode=io_mode)
            eeg = np.random.randn(32, 128)
            io.write_eeg(eeg, key='0')
            eeg = np.random.randn(32, 128)
            io.write_eeg(eeg, key='1')
            self.assertEqual(len(io), 2)

    def test_read_eeg(self):
        io_io_path = os.path.join(self._io_path, 'signals')
        io_mode_space = ['pickle', 'lmdb']
        for io_mode in io_mode_space:
            io = _make_backend(io_path=os.path.join(
                io_io_path, io_mode), io_mode=io_mode)
            eeg = np.random.randn(32, 128)
            io_eeg_index = io.write_eeg(eeg, key='0')
            io_eeg = io.read_eeg(io_eeg_index)
            self.assertTrue(np.array_equal(eeg, io_eeg))

    def test_write_eeg_of_different_types(self):
        io_io_path = os.path.join(self._io_path, 'signals')
        io_mode_space = ['pickle', 'lmdb']
        for io_mode in io_mode_space:
            io = _make_backend(io_path=os.path.join(
                io_io_path, io_mode), io_mode=io_mode)
            eeg = torch.randn(32, 128)
            io_eeg_index = io.write_eeg(eeg, key='0')
            eeg = np.random.randn(32, 128)
            io_eeg_index = io.write_eeg(eeg, key='1')
            io_eeg = io.read_eeg(io_eeg_index)
            self.assertTrue(np.array_equal(eeg, io_eeg))

    def test_write_eeg_of_different_shapes(self):
        io_io_path = os.path.join(self._io_path, 'signals')
        io_mode_space = ['pickle', 'lmdb']
        for io_mode in io_mode_space:
            io = _make_backend(io_path=os.path.join(
                io_io_path, io_mode), io_mode=io_mode)
            eeg = np.random.randn(32, 128)
            io_eeg_index = io.write_eeg(eeg, key='0')
            eeg = np.random.randn(128, 9, 9)
            io_eeg_index = io.write_eeg(eeg, key='1')
            io_eeg = io.read_eeg(io_eeg_index)
            self.assertTrue(np.array_equal(eeg, io_eeg))

    def test_write_eeg_requires_string_key(self):
        for mode in ('memory', 'pickle', 'lmdb'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as path:
                io = _make_backend(path, io_mode=mode)
                sample = np.array([10])
                with self.assertRaises(TypeError):
                    io.write_eeg(sample)
                for key in (None, 0):
                    with self.assertRaisesRegex(TypeError, 'key must be a string'):
                        io.write_eeg(sample, key=key)
                    with self.assertRaisesRegex(TypeError, 'key must be a string'):
                        io.write_eeg(sample, key=key)
                self.assertEqual(len(io), 0)
                del io

    def test_write_eeg_preserves_other_keys_and_allows_replacement(self):
        for mode in ('memory', 'pickle', 'lmdb'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as path:
                io = _make_backend(path, io_mode=mode)
                io.write_eeg(np.array([10]), key='0')
                io.write_eeg(np.array([20]), key='2')
                self.assertEqual(io.write_eeg(
                    np.array([30]), key='new'), 'new')
                io.write_eeg(np.array([22]), key='2')
                self.assertEqual(len(io), 3)
                for key, value in (('0', 10), ('2', 22), ('new', 30)):
                    np.testing.assert_array_equal(io.read_eeg(key), [value])
                del io
