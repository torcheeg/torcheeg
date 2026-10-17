import os
import unittest
import tempfile

from torcheeg.io import MetaInfoIO


class TestMetaInfoIO(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.io_path = self.directory.name

    def test_init(self):
        io_io_path = os.path.join(self.io_path, 'info.csv')
        io = MetaInfoIO(io_path=io_io_path)
        self.assertEqual(len(io), 0)

        self.assertTrue(os.path.exists(io_io_path))
        io = MetaInfoIO(io_path=io_io_path)
        self.assertEqual(len(io), 0)

    def test_write_info(self):
        io_io_path = os.path.join(self.io_path, 'signals')
        io = MetaInfoIO(io_path=io_io_path)

        io.write_info({'subject_id': 0, 'clip_id': 0, 'baseline_id': 0, 'valence': 0, 'arousal': 0})

        io.write_info({'subject_id': 0, 'clip_id': 1, 'baseline_id': 0, 'valence': 0, 'arousal': 0})

    def test_len(self):
        io_io_path = os.path.join(self.io_path, 'signals')
        io = MetaInfoIO(io_path=io_io_path)

        io.write_info({'subject_id': 0, 'clip_id': 0, 'baseline_id': 0, 'valence': 0, 'arousal': 0})

        io.write_info({'subject_id': 0, 'clip_id': 1, 'baseline_id': 0, 'valence': 0, 'arousal': 0})
        self.assertEqual(len(io), 2)

    def test_read_info(self):
        io_io_path = os.path.join(self.io_path, 'signals')
        io = MetaInfoIO(io_path=io_io_path)

        info = {'subject_id': 0, 'clip_id': 1, 'baseline_id': 0, 'valence': 0, 'arousal': 0}
        io_info_index = io.write_info(info)
        io_info = io.read_info(io_info_index)
        for key in info.keys():
            self.assertEqual(io_info[key], info[key])

    def test_read_all(self):
        io_io_path = os.path.join(self.io_path, 'signals')
        io = MetaInfoIO(io_path=io_io_path)

        io.write_info({'subject_id': 0, 'clip_id': 0, 'baseline_id': 0, 'valence': 0, 'arousal': 0})

        io.write_info({'subject_id': 0, 'clip_id': 1, 'baseline_id': 0, 'valence': 0, 'arousal': 0})
        self.assertEqual(len(io.read_all()), 2)


if __name__ == '__main__':
    unittest.main()