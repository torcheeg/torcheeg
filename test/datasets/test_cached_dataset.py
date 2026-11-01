import inspect
import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from torcheeg.datasets import BaseData, DEAPData, DEAPDataset


def _fake_data(dataset_class, **configuration):
    cls = dataset_class.data_class
    defaults = {name: p.default for name, p in inspect.signature(
        cls).parameters.items() if p.default is not inspect.Parameter.empty}
    defaults.update(configuration)
    data = cls.__new__(cls)
    BaseData.__init__(data, **defaults)
    return data


def _process_fake(dataset_class, *args, **kwargs):
    return _fake_data(dataset_class, **kwargs).process_record(*args, **kwargs)


def _assert_memory_dataset(test_case, dataset_class, **configuration):
    data = _fake_data(dataset_class, **configuration)
    payload = data.fake_record(record='fake_record')
    record = payload.pop('record', 'fake_record')
    expected = next(data.process_record(record=record, **payload))
    data_class = dataset_class.data_class

    def one_sample(self, **kwargs):
        yield expected

    def increment(sample):
        return {**sample, 'eeg': sample['eeg'] + 1}
    with patch.object(data_class, 'set_records', return_value=[record]), patch.object(data_class, 'read_record', return_value=payload), patch.object(data_class, 'process_record', one_sample):
        dataset = dataset_class(
            io_mode='memory', verbose=False, transform=increment, **configuration)
    test_case.assertEqual(len(dataset), 1)
    np.testing.assert_array_equal(dataset[0]['eeg'], expected['eeg'] + 1)
    changed = dataset.with_transform(
        lambda sample: {**sample, 'eeg': sample['eeg'] + 3})
    np.testing.assert_array_equal(changed[0]['eeg'], expected['eeg'] + 1 + 3)
    test_case.assertNotIn('clip_id', dataset[0]['info'])


class TestCachedDataset(unittest.TestCase):

    def make_source(self, directory):
        source = os.path.join(directory, 'raw')
        os.makedirs(source)
        signal = np.arange(32, dtype=np.float32).reshape(1, 2, 16)
        with open(os.path.join(source, 's01.dat'), 'wb') as file:
            pickle.dump(
                {'data': signal, 'labels': np.array([[1, 2, 3, 4]])}, file)
        return dict(root_path=source, chunk_size=4, overlap=2, num_channel=2, num_baseline=1, baseline_chunk_size=4)

    def test_persistent_reuse_and_changed_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_source(directory)
            path = os.path.join(directory, 'cache')
            dataset = DEAPDataset(**config, io_path=path)
            expected = dataset[0]['eeg'].copy()
            self.assertEqual(len(dataset), 5)
            dataset.close()
            dataset = DEAPDataset(**config, io_path=path)
            np.testing.assert_array_equal(dataset[0]['eeg'], expected)
            dataset.close()
            config['overlap'] = 0
            with self.assertRaises(ValueError):
                DEAPDataset(**config, io_path=path)

    def test_failure_does_not_publish_manifest_and_retry_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_source(directory)
            path = os.path.join(directory, 'cache')

            def fail(sample):
                raise RuntimeError('deliberate failure')
            with self.assertRaisesRegex(RuntimeError, 'deliberate failure'):
                DEAPDataset(**config, io_path=path, transform=fail)
            self.assertFalse(os.path.exists(
                os.path.join(path, 'manifest.json')))
            dataset = DEAPDataset(**config, io_path=path)
            self.assertEqual(len(dataset), 5)
            dataset.close()
