import os
import random
import unittest
import shutil

from torcheeg.io import MetaInfoIO


class TestMetaInfoIO(unittest.TestCase):
    def setUp(self):
        shutil.rmtree('./outputs/')
        os.mkdir('./outputs/')

    def test_init(self):
        io_cache_path = f'./outputs/{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}.csv'
        io = MetaInfoIO(cache_path=io_cache_path)
        self.assertEqual(len(io), 0)

        self.assertTrue(os.path.exists(io_cache_path))
        io = MetaInfoIO(cache_path=io_cache_path)
        self.assertEqual(len(io), 0)

    def test_write_info(self):
        io_cache_path = f'./outputs/{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        io = MetaInfoIO(cache_path=io_cache_path)

        io.write_info({'subject': 0, 'clip_id': 0, 'baseline_id': 0, 'valence': 0, 'arousal': 0})

        io.write_info({'subject': 0, 'clip_id': 1, 'baseline_id': 0, 'valence': 0, 'arousal': 0})

    def test_len(self):
        io_cache_path = f'./outputs/{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        io = MetaInfoIO(cache_path=io_cache_path)

        io.write_info({'subject': 0, 'clip_id': 0, 'baseline_id': 0, 'valence': 0, 'arousal': 0})

        io.write_info({'subject': 0, 'clip_id': 1, 'baseline_id': 0, 'valence': 0, 'arousal': 0})
        self.assertEqual(len(io), 2)

    def test_read_info(self):
        io_cache_path = f'./outputs/{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        io = MetaInfoIO(cache_path=io_cache_path)

        info = {'subject': 0, 'clip_id': 1, 'baseline_id': 0, 'valence': 0, 'arousal': 0}
        io_info_idx = io.write_info(info)
        io_info = io.read_info(io_info_idx)
        for key in info.keys():
            self.assertEqual(io_info[key], info[key])

    def test_read_all(self):
        io_cache_path = f'./outputs/{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        io = MetaInfoIO(cache_path=io_cache_path)

        io.write_info({'subject': 0, 'clip_id': 0, 'baseline_id': 0, 'valence': 0, 'arousal': 0})

        io.write_info({'subject': 0, 'clip_id': 1, 'baseline_id': 0, 'valence': 0, 'arousal': 0})
        self.assertEqual(len(io.read_all()), 2)


if __name__ == '__main__':
    unittest.main()