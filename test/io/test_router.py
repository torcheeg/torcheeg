import os
import random
import shutil
import unittest

import numpy as np
import pandas as pd

from torcheeg.io import EEGSignalIO, IORouter, LazyIORouter


class TestIORouter(unittest.TestCase):
    def setUp(self):
        if os.path.exists('./tmp_out/'):
            shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def fake_io(self, io_path):
        os.makedirs(io_path)

        records = ['_record_0', '_record_1']
        test_data = {}

        for record in records:
            record_path = os.path.join(io_path, record)
            os.makedirs(record_path)
            os.makedirs(os.path.join(record_path, 'eeg'))

            info_data = {
                'clip_id': [0, 1],
                'duration': [10, 20],
                'label': ['A', 'B']
            }
            info_df = pd.DataFrame(info_data)
            info_df.to_csv(os.path.join(record_path, 'info.csv'), index=False)

            eeg_io = EEGSignalIO(str(os.path.join(record_path, 'eeg')))
            for i in range(2):
                data = np.ones([100, 32]) * i
                eeg_io.write_eeg(data, str(i))
                test_data[f"{record}_{i}"] = data

    def test_init(self):
        io_path = f'./tmp_out/{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        self.fake_io(io_path)

        # Test both routers can be initialized
        io_router = IORouter(io_path)
        lazy_router = LazyIORouter(io_path)

        # Check if both routers have same number of samples
        assert len(io_router) == len(lazy_router)
        assert len(io_router) == 4  # 2 records × 2 samples

    def test_read_info(self):
        io_path = f'./tmp_out/{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        self.fake_io(io_path)

        io_router = IORouter(io_path)
        lazy_router = LazyIORouter(io_path)

        # Test if both routers return same info
        for i in range(len(io_router)):
            info1 = io_router.read_info(i)
            info2 = lazy_router.read_info(i)
            assert info1 == info2

    def test_read_eeg(self):
        io_path = f'./tmp_out/{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        self.fake_io(io_path)

        io_router = IORouter(io_path)
        lazy_router = LazyIORouter(io_path)

        # Test reading EEG data
        for record in ['_record_0', '_record_1']:
            for key in ['0', '1']:
                data1 = io_router.read_eeg(record, key)
                data2 = lazy_router.read_eeg(record, key)

                assert np.array_equal(data1, data2)

    def test_write_eeg(self):
        io_path = f'./tmp_out/{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        self.fake_io(io_path)

        io_router = IORouter(io_path)
        lazy_router = LazyIORouter(io_path)

        # Test writing new EEG data
        new_data = np.random.randn(100, 32)
        record = '_record_0'
        key = '0'

        # Write using both routers
        io_router.write_eeg(record, key, new_data)
        read_data1 = io_router.read_eeg(record, key)
        assert np.array_equal(read_data1, new_data)

        lazy_router.write_eeg(record, key, new_data)
        read_data2 = lazy_router.read_eeg(record, key)
        assert np.array_equal(read_data2, new_data)

    def test_getitem(self):
        io_path = f'./tmp_out/{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        self.fake_io(io_path)

        io_router = IORouter(io_path)
        lazy_router = LazyIORouter(io_path)

        # Test __getitem__ returns same results
        for i in range(len(io_router)):
            eeg1, info1 = io_router[i]
            eeg2, info2 = lazy_router[i]

            assert np.array_equal(eeg1, eeg2)
            assert info1 == info2

    def test_copy(self):
        io_path = f'./tmp_out/{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        self.fake_io(io_path)

        io_router = IORouter(io_path)
        lazy_router = LazyIORouter(io_path)

        # Test copy functionality
        io_router_copy = io_router.__copy__()
        lazy_router_copy = lazy_router.__copy__()

        # Verify copies work the same as originals
        assert len(io_router_copy) == len(io_router)
        assert len(lazy_router_copy) == len(lazy_router)

        # Test reading from copies
        eeg1, info1 = io_router[0]
        eeg2, info2 = io_router_copy[0]
        assert np.array_equal(eeg1, eeg2)
        assert info1 == info2

        eeg3, info3 = lazy_router[0]
        eeg4, info4 = lazy_router_copy[0]
        assert np.array_equal(eeg3, eeg4)
        assert info3 == info4


if __name__ == '__main__':
    unittest.main()
