import os
import tempfile
import unittest
from test.datasets.test_cached_dataset import (_assert_memory_dataset,
                                               _process_fake)

import numpy as np

from torcheeg.datasets import DEAPDataset


class TestInputValidation(unittest.TestCase):

    def test_missing_dataset_directory_is_a_file_error(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(FileNotFoundError):
                DEAPDataset(root_path=os.path.join(path, 'missing'),
                            io_path=os.path.join(path, 'cache'), verbose=False)


class TestDEAPDataset(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_deap_dataset(self):
        kwargs = {'record': 'fake_record'}
        kwargs.update(DEAPDataset.data_class.fake_record(**kwargs))
        item = next(_process_fake(DEAPDataset, **kwargs))
        self.assertEqual(item['eeg'].shape, (32, 128))


class TestFakeRecordMetadata(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_fake_samples_have_consistent_keys_and_metadata(self):
        for dataset_class in (DEAPDataset,):
            with self.subTest(dataset=dataset_class.__name__):
                kwargs = {'record': 'fake_record'}
                kwargs.update(dataset_class.data_class.fake_record(**kwargs))
                sample = next((item for item in _process_fake(
                    dataset_class, **kwargs) if 'info' in item))
                self.assertIsInstance(sample['info'], dict)
                self.assertNotIn('key', sample)
                self.assertNotIn('clip_id', sample['info'])
                self.assertGreater(sample['eeg'].size, 0)
                self.assertTrue(np.isfinite(sample['eeg']).all())

    def test_memory_construction_preserves_preprocessing_and_item_transforms(self):
        for dataset_class in (DEAPDataset,):
            with self.subTest(dataset=dataset_class.__name__):
                _assert_memory_dataset(self, dataset_class)


class TestDEAPData(unittest.TestCase):

    def test_instance_configuration_and_default_dataset_are_equivalent(self):
        import os
        import pickle
        import tempfile

        from torcheeg.datasets import DEAPData, IODataset
        from torcheeg.io import MemoryEEGIO, MemoryInfoIO
        from torcheeg.runners import SequentialRunner

        def increment(sample):
            return {**sample, 'eeg': sample['eeg'] + 1, 'baseline': sample['baseline'] + 1}
        with tempfile.TemporaryDirectory() as root:
            samples = np.arange(2 * 16).reshape(1, 2, 16).astype(np.float32)
            labels = np.array([[1, 2, 3, 4]])
            with open(os.path.join(root, 's01.dat'), 'wb') as file:
                pickle.dump({'data': samples, 'labels': labels}, file)
            configuration = dict(root_path=root, chunk_size=4, overlap=2,
                                 num_channel=2, num_baseline=1, baseline_chunk_size=4)
            data = DEAPData(**configuration, transform=increment)
            self.assertEqual(data.set_records(), ['s01.dat'])
            payload = data.read_record('s01.dat')
            self.assertEqual(
                len(list(data.process_record('s01.dat', **payload))), 5)
            eeg, info = (MemoryEEGIO(), MemoryInfoIO())
            IODataset(eeg, info).extend(SequentialRunner()(data))
            explicit = IODataset(eeg, info)
            shortcut = DEAPDataset(
                **configuration, io_mode='memory', offline_transform=increment, verbose=False)
            self.assertEqual(len(explicit), 5)
            self.assertEqual(len(shortcut), len(explicit))
            for index in range(len(explicit)):
                np.testing.assert_array_equal(
                    explicit[index]['eeg'], shortcut[index]['eeg'])
                self.assertEqual(
                    explicit[index]['info'], shortcut[index]['info'])
            self.assertEqual(explicit[0]['info']['valence'], 1)
            np.testing.assert_array_equal(
                explicit[0]['eeg'], samples[0, :, 4:8] + 1)

    def test_default_shortcut_reuses_each_persistent_backend(self):
        import os
        import pickle
        import tempfile
        modes = ['lmdb', 'pickle', 'numpy', 'zarr']
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, 'source')
            os.mkdir(source)
            samples = np.arange(32, dtype=np.float32).reshape(1, 2, 16)
            with open(os.path.join(source, 's01.dat'), 'wb') as file:
                pickle.dump(
                    {'data': samples, 'labels': np.array([[1, 2, 3, 4]])}, file)
            for mode in modes:
                with self.subTest(mode=mode):
                    configuration = dict(root_path=source, chunk_size=4, overlap=2, num_channel=2, num_baseline=1,
                                         baseline_chunk_size=4, io_mode=mode, io_path=os.path.join(root, mode), verbose=False)
                    first = DEAPDataset(**configuration)
                    repeated = DEAPDataset(**configuration, num_worker=2)
                    self.assertEqual(len(first), 5)
                    self.assertEqual(len(repeated), 5)
                    np.testing.assert_array_equal(
                        first[0]['eeg'], samples[0, :, 4:8])
                    np.testing.assert_array_equal(
                        repeated[0]['eeg'], first[0]['eeg'])
                    self.assertEqual(first[0]['info']['valence'], 1)
                    first.eeg_io.close()
                    repeated.eeg_io.close()


if __name__ == '__main__':
    unittest.main()
