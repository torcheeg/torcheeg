import os
import tempfile
import unittest
from test.io.info_io.test_base import _assert_backend

from torcheeg.io import CSVInfoIO


class TestCSVInfoIO(unittest.TestCase):
    def test_backend_contract(self):
        _assert_backend(self, CSVInfoIO)

    def test_optional_fields_are_unioned_without_changing_existing_values(self):
        import tempfile
        from pathlib import Path

        import pandas as pd

        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root, 'info.csv'))
            io = CSVInfoIO(path)
            io.write_info({'clip_id': '001', 'subject_id': 'NA'})
            io.write_info({'clip_id': '002', 'note': 'second'})
            io.write_info({'clip_id': '003', 'subject_id': '03'})
            rows = CSVInfoIO(path).read_all()
            self.assertEqual(rows['clip_id'].tolist(), ['001', '002', '003'])
            self.assertEqual(rows.loc[0, 'subject_id'], 'NA')
            self.assertEqual(rows.loc[1, 'note'], 'second')
            self.assertTrue(pd.isna(rows.loc[0, 'note']))
            self.assertTrue(pd.isna(rows.loc[1, 'subject_id']))
            before = Path(path).read_bytes()
            with self.assertRaises(TypeError):
                io.write_info({'clip_id': '004', 'unsupported': object()})
            self.assertEqual(before, Path(path).read_bytes())


class TestCSVMetadataRows(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._io_path = self._directory.name

    def test_init(self):
        io_io_path = os.path.join(self._io_path, 'info.csv')
        io = CSVInfoIO(io_path=io_io_path)
        self.assertEqual(len(io), 0)

        self.assertTrue(os.path.exists(io_io_path))
        io = CSVInfoIO(io_path=io_io_path)
        self.assertEqual(len(io), 0)

    def test_write_info(self):
        io_io_path = os.path.join(self._io_path, 'signals')
        io = CSVInfoIO(io_path=io_io_path)

        io.write_info({'subject_id': 0, 'clip_id': 0,
                      'baseline_id': 0, 'valence': 0, 'arousal': 0})

        io.write_info({'subject_id': 0, 'clip_id': 1,
                      'baseline_id': 0, 'valence': 0, 'arousal': 0})

        self.assertEqual(len(io), 2)
        self.assertEqual(io.read_info(0)['clip_id'], 0)
        self.assertEqual(io.read_info(1)['clip_id'], 1)

    def test_len(self):
        io_io_path = os.path.join(self._io_path, 'signals')
        io = CSVInfoIO(io_path=io_io_path)

        io.write_info({'subject_id': 0, 'clip_id': 0,
                      'baseline_id': 0, 'valence': 0, 'arousal': 0})

        io.write_info({'subject_id': 0, 'clip_id': 1,
                      'baseline_id': 0, 'valence': 0, 'arousal': 0})
        self.assertEqual(len(io), 2)

    def test_read_info(self):
        io_io_path = os.path.join(self._io_path, 'signals')
        io = CSVInfoIO(io_path=io_io_path)

        info = {'subject_id': 0, 'clip_id': 1,
                'baseline_id': 0, 'valence': 0, 'arousal': 0}
        io_info_index = io.write_info(info)
        io_info = io.read_info(io_info_index)
        for key in info.keys():
            self.assertEqual(io_info[key], info[key])

    def test_read_all(self):
        io_io_path = os.path.join(self._io_path, 'signals')
        io = CSVInfoIO(io_path=io_io_path)

        io.write_info({'subject_id': 0, 'clip_id': 0,
                      'baseline_id': 0, 'valence': 0, 'arousal': 0})

        io.write_info({'subject_id': 0, 'clip_id': 1,
                      'baseline_id': 0, 'valence': 0, 'arousal': 0})
        self.assertEqual(len(io.read_all()), 2)


if __name__ == '__main__':
    unittest.main()
