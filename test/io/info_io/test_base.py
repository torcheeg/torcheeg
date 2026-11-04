import tempfile
import unittest
from pathlib import Path

import numpy as np

from torcheeg.io import BaseInfoIO


def _assert_backend(test, backend):
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory, 'info'))
        io = backend(path)
        row = {'clip_id': '001', 'subject_id': 'NA', 'trial_id': 1,
               'flag': True, 'score': np.float32(1.5),
               'extra': {'values': [1, None]}, 'missing': None}
        test.assertEqual(len(io), 0)
        test.assertEqual(io.write_info(row), 0)
        row['clip_id'] = '002'
        test.assertEqual(io.write_info(row), 1)
        test.assertEqual(len(io), 2)
        test.assertEqual(io.read_info(0)['clip_id'], '001')
        test.assertEqual(io.read_info(0)['subject_id'], 'NA')
        test.assertEqual(io.read_info(0)['extra'], {'values': [1, None]})
        test.assertIsInstance(io.read_info(0)['trial_id'], (int, np.integer))
        test.assertEqual(io.read_info(0)['score'], 1.5)
        test.assertIsNone(io.read_info(0)['missing'])
        with test.assertRaises(TypeError):
            io.write_info([])
        if io.persistent:
            reopened = backend(path)
            test.assertEqual(len(reopened), 2)
            test.assertEqual(reopened.read_info(0)['clip_id'], '001')
        test.assertEqual(io[0]['clip_id'], '001')
        test.assertEqual(io[-1]['clip_id'], '002')
        test.assertEqual(io['clip_id'].eq('002').tolist(), [False, True])
        test.assertEqual(io[:1]['clip_id'].tolist(), ['001'])
        test.assertEqual(io[[1, 0, 1]]['clip_id'].tolist(),
                         ['002', '001', '002'])
        test.assertEqual(io[np.array([False, True])]
                         ['clip_id'].tolist(), ['002'])
        test.assertEqual(len(io[[]]), 0)
        for index, error in (([True], ValueError), ([[0]], ValueError),
                             ([1.0], TypeError), (['0'], TypeError)):
            with test.assertRaises(error):
                io[index]
        clone = io.clone(str(Path(directory, 'clone')))
        test.assertEqual(len(clone), 0)
        io.close()


class TestBaseInfoIO(unittest.TestCase):
    def test_required_methods_identify_the_class(self):
        io = BaseInfoIO('unused')
        for method, args in (('read_all', ()), ('write_info', ({},))):
            with self.subTest(method=method), self.assertRaisesRegex(
                    NotImplementedError, 'Method ' + method + ' is not implemented in class BaseInfoIO'):
                getattr(io, method)(*args)
        io.close()
