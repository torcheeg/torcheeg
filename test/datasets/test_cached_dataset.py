import inspect
import os
import pickle
import tempfile
import unittest
from copy import copy
from unittest.mock import patch

import numpy as np

from torcheeg.datasets import BaseData, DEAPDataset


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

    def increment(eeg, baseline=None, **kwargs):
        return {'eeg': eeg + 1}
    with patch.object(data_class, 'set_records', return_value=[record]), patch.object(data_class, 'read_record', return_value=payload), patch.object(data_class, 'process_record', one_sample):
        dataset = dataset_class(
            io_mode='memory', verbose=False, offline_transform=increment,
            online_transform=lambda eeg, **kwargs: {'eeg': eeg * 2}, **configuration)
    test_case.assertEqual(len(dataset), 1)
    np.testing.assert_array_equal(dataset[0]['eeg'], (expected['eeg'] + 1) * 2)
    changed = copy(dataset)
    changed.transform = lambda eeg, **kwargs: {'eeg': eeg + 3}
    np.testing.assert_array_equal(
        changed[0]['eeg'], expected['eeg'] + 1 + 3)
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

            def fail(eeg, **kwargs):
                raise RuntimeError('deliberate failure')
            with self.assertRaisesRegex(RuntimeError, 'deliberate failure'):
                DEAPDataset(**config, io_path=path, offline_transform=fail)
            self.assertFalse(os.path.exists(
                os.path.join(path, 'manifest.json')))
            dataset = DEAPDataset(**config, io_path=path)
            self.assertEqual(len(dataset), 5)
            dataset.close()

    def test_online_transform_runs_on_each_read_and_reuses_offline_cache(self):
        def offline(eeg, **kwargs):
            return {'eeg': eeg + 10}

        calls = []

        def online(eeg, baseline=None, **kwargs):
            calls.append(1)
            eeg *= 2
            return {'eeg': eeg, 'baseline': None}

        def label(info, **kwargs):
            return {'label': info['valence']}

        with tempfile.TemporaryDirectory() as directory:
            config = self.make_source(directory)
            path = os.path.join(directory, 'cache')
            dataset = DEAPDataset(**config, io_path=path,
                                  offline_transform=offline, online_transform=online, label_transform=label)
            self.assertEqual(calls, [])
            first = dataset[0]
            np.testing.assert_array_equal(dataset[0]['eeg'], first['eeg'])
            self.assertEqual(calls, [1, 1])
            self.assertEqual(set(first), {'eeg', 'label'})
            self.assertEqual(first['label'], 1)
            dataset.close()
            with patch('torcheeg.datasets.cached_dataset.MultiprocessRunner.iterate', side_effect=AssertionError('cache rebuilt')):
                dataset = DEAPDataset(
                    **config, io_path=path, offline_transform=offline)
                np.testing.assert_array_equal(
                    dataset[0]['eeg'] * 2, first['eeg'])
                dataset.close()
            with self.assertRaises(ValueError):
                DEAPDataset(**config, io_path=path)

    def test_stage_transforms_must_be_callable(self):
        for name in ('offline_transform', 'online_transform', 'label_transform'):
            with self.subTest(name=name), self.assertRaisesRegex(TypeError, name):
                DEAPDataset(io_mode='memory', **{name: 1})

    def test_offline_baseline_removal_is_persisted_as_absence(self):
        def offline(eeg, baseline=None, **kwargs):
            return {'eeg': eeg - baseline, 'baseline': None}

        seen = []

        def online(eeg, baseline=None, **kwargs):
            seen.append(baseline)
            return {'eeg': eeg, 'baseline': baseline}

        with tempfile.TemporaryDirectory() as directory:
            config = self.make_source(directory)
            path = os.path.join(directory, 'cache')
            dataset = DEAPDataset(**config, io_path=path,
                                  offline_transform=offline)
            self.assertNotIn('baseline_id', dataset.info)
            expected = dataset[0]['eeg']
            dataset.close()
            dataset = DEAPDataset(**config, io_path=path, offline_transform=offline,
                                  online_transform=online,
                                  label_transform=lambda info, **kwargs: {'label': info['valence']})
            sample = dataset[0]
            self.assertEqual(seen, [None])
            self.assertEqual(set(sample), {'eeg', 'label'})
            np.testing.assert_array_equal(sample['eeg'], expected)
            dataset.close()
