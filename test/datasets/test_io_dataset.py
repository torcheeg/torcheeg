import pickle
import tempfile
import unittest
from copy import copy
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from torcheeg.datasets import IODataset
from torcheeg.io import CSVInfoIO, MemoryEEGIO, MemoryInfoIO, NumpyEEGIO
from torcheeg.transforms import Compose


def add_one(eeg, baseline=None, **kwargs):
    eeg += 1
    return {'eeg': eeg}


def increment_array(eeg):
    return eeg + 1


def double_array(eeg):
    return eeg * 2


def compose_signals(eeg, baseline=None, **kwargs):
    transform = Compose([increment_array, Compose([double_array])])
    return {'eeg': transform(eeg), 'baseline': None}


def prepare_label(info, **kwargs):
    return {'label': info['label']}


def drop_baseline(eeg, baseline=None, **kwargs):
    return {'eeg': eeg, 'baseline': None}


class TestIODataset(unittest.TestCase):

    def make_dataset(self, transform=None, label_transform=None):
        dataset = IODataset(MemoryEEGIO(), MemoryInfoIO(
        ), transform=transform, label_transform=label_transform)
        for index in range(3):
            sample = {'eeg': np.full((2, 4), index, dtype=np.float32), 'info': {
                'subject_id': f'{index:03}', 'label': index}}
            if index != 1:
                sample['baseline'] = np.ones((2, 4), dtype=np.float32)
            dataset.append(sample)
        return dataset

    def test_dictionary_roundtrip_and_baseline(self):
        dataset = self.make_dataset()
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0]['info'], {'subject_id': '000', 'label': 0})
        self.assertEqual(set(dataset[0]), {'eeg', 'info', 'baseline'})
        self.assertNotIn('baseline', dataset[1])
        self.assertEqual([s['info']['label'] for s in dataset], [0, 1, 2])
        self.assertEqual(dataset[-1]['info']['label'], 2)
        with self.assertRaises(IndexError):
            dataset[3]

    def test_selection_order_duplicates_and_empty(self):
        dataset = self.make_dataset()
        for indices, expected in ((slice(None, None, -1), [2, 1, 0]), ([2, 0, 2], [2, 0, 2]), ([], []), ([True, False, True], [0, 2])):
            selected = dataset[indices]
            self.assertIs(selected.eeg_io, dataset.eeg_io)
            self.assertEqual([s['info']['label'] for s in selected], expected)
        mask = dataset.info['subject_id'].eq('001').to_numpy()
        self.assertEqual(dataset[mask][0]['info']['label'], 1)

    def test_invalid_indices(self):
        dataset = self.make_dataset()
        for index, error in ((True, TypeError), ([True], ValueError), ([1.0], TypeError), ([[1]], ValueError), (['0'], TypeError)):
            with self.subTest(index=index), self.assertRaises(error):
                dataset[index]

    def test_append_and_read_do_not_alias_signals(self):
        dataset = IODataset(MemoryEEGIO(), MemoryInfoIO())
        eeg = np.ones((2, 4))
        sample = {'eeg': eeg, 'baseline': eeg, 'info': {'subject_id': '001'}}
        dataset.append(sample)
        eeg[:] = 9
        sample['info']['subject_id'] = 'changed'
        result = dataset[0]
        result['eeg'][:] = 7
        result['baseline'][:] = 8
        result['info']['subject_id'] = 'edited'
        np.testing.assert_array_equal(dataset[0]['eeg'], np.ones((2, 4)))
        np.testing.assert_array_equal(dataset[0]['baseline'], np.ones((2, 4)))
        self.assertEqual(dataset[0]['info']['subject_id'], '001')

    def test_transform_composition_selection_and_serialization(self):
        dataset = self.make_dataset()
        chain = IODataset(dataset.eeg_io, dataset.info_io,
                          transform=compose_signals)
        composed = IODataset(dataset.eeg_io, dataset.info_io,
                             transform=compose_signals)
        for index in range(3):
            np.testing.assert_array_equal(
                chain[index]['eeg'], composed[index]['eeg'])
            np.testing.assert_array_equal(
                chain[index]['eeg'], np.full((2, 4), 2 * (index + 1)))
        np.testing.assert_array_equal(dataset[0]['eeg'], np.zeros((2, 4)))
        np.testing.assert_array_equal(chain[[2, 0]][0]['eeg'], chain[2]['eeg'])
        restored = pickle.loads(pickle.dumps(chain))
        np.testing.assert_array_equal(restored[0]['eeg'], chain[0]['eeg'])

    def test_views_cannot_append_and_metadata_is_independent(self):
        dataset = self.make_dataset()
        for view in (dataset[:1], copy(dataset)):
            view.info.loc[0, 'subject_id'] = 'edited'
            self.assertEqual(dataset.info.loc[0, 'subject_id'], '000')
            with self.assertRaises(ValueError):
                view.append(dataset[0])

    def test_extend_streams_and_closes_failed_producer(self):
        dataset = IODataset(MemoryEEGIO(), MemoryInfoIO())
        closed = []

        def samples():
            try:
                for index in range(3):
                    self.assertEqual(len(dataset), index)
                    yield {'eeg': np.array([index]), 'info': {'label': index}}
                yield {'info': {}}
            finally:
                closed.append(True)
        with self.assertRaises(ValueError):
            dataset.extend(samples())
        self.assertEqual(len(dataset), 3)
        self.assertEqual(closed, [True])

    def test_transformed_samples_can_be_saved(self):
        source = self.make_dataset()
        target = IODataset(MemoryEEGIO(), MemoryInfoIO())
        target.extend(
            IODataset(source.eeg_io, source.info_io, transform=add_one))
        target.extend(source[:1])
        self.assertEqual(len(target), 4)
        np.testing.assert_array_equal(target[0]['eeg'], np.ones((2, 4)))
        np.testing.assert_array_equal(target[-1]['eeg'], np.zeros((2, 4)))
        self.assertEqual(len(set(target.info['_record_id'])), 4)

    def test_persistent_roundtrip_preserves_string_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            eeg_path = str(Path(directory) / 'eeg')
            info_path = str(Path(directory) / 'info.csv')
            dataset = IODataset(NumpyEEGIO(eeg_path), CSVInfoIO(info_path))
            dataset.extend(self.make_dataset())
            dataset.close()
            reopened = IODataset(NumpyEEGIO(eeg_path), CSVInfoIO(info_path))
            self.assertEqual(reopened[0]['info']['subject_id'], '000')
            changed = IODataset(
                reopened.eeg_io, reopened.info_io, transform=add_one)[0]
            np.testing.assert_array_equal(changed['eeg'], np.ones((2, 4)))
            np.testing.assert_array_equal(reopened[0]['eeg'], np.zeros((2, 4)))
            reopened.close()

    def test_training_batch_is_dictionary(self):
        dataset = self.make_dataset(
            transform=drop_baseline, label_transform=prepare_label)
        batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
        self.assertEqual(set(batch), {'eeg', 'label'})
        self.assertEqual(tuple(batch['eeg'].shape), (2, 2, 4))
        self.assertEqual(batch['label'].tolist(), [0, 1])

    def test_spawned_loader_and_cpu_training(self):
        import torch
        dataset = self.make_dataset(
            transform=drop_baseline, label_transform=prepare_label)
        loader = DataLoader(dataset, batch_size=2,
                            num_workers=1, multiprocessing_context='spawn')
        batches = list(loader)
        self.assertEqual([batch['label'].tolist()
                         for batch in batches], [[0, 1], [2]])
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8, 3))
        loss = torch.nn.functional.cross_entropy(
            model(batches[0]['eeg']), batches[0]['label'])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model[1].weight.grad)

    def test_invalid_sample_and_transform_contracts(self):
        dataset = self.make_dataset()
        for sample in (None, {'eeg': None, 'info': {}}, {'eeg': np.zeros(1), 'info': []}, {'eeg': np.zeros(1), 'info': {'clip_id': 'external'}}, {'eeg': np.zeros(1), 'info': {}, 'label': 1}):
            with self.subTest(sample=sample), self.assertRaises((TypeError, ValueError)):
                dataset.append(sample)
        with self.assertRaises(TypeError):
            IODataset(dataset.eeg_io, dataset.info_io, transform=1)
        with self.assertRaises(TypeError):
            IODataset(dataset.eeg_io, dataset.info_io,
                      transform=lambda eeg, **kwargs: eeg)[0]

    def test_none_baseline_is_not_stored_and_collates_without_custom_loader(self):
        calls = []

        def signal(eeg, baseline=None, **kwargs):
            calls.append(baseline)
            return {'eeg': eeg + 1, 'baseline': None, 'unused': None}

        dataset = IODataset(MemoryEEGIO(), MemoryInfoIO(),
                            transform=signal, label_transform=prepare_label)
        dataset.extend({'eeg': np.zeros((2, 4)), 'baseline': None,
                        'info': {'label': i}} for i in range(2))
        self.assertEqual(len(dataset.eeg_io), 2)
        self.assertNotIn('baseline_id', dataset.info)
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        self.assertEqual(set(batch), {'eeg', 'label'})
        self.assertEqual(calls, [None, None])
        np.testing.assert_array_equal(batch['eeg'].numpy(), np.ones((2, 2, 4)))

    def test_baseline_roundtrip_removal_and_extra_batch_fields(self):
        from torcheeg.transforms import BaselineRemoval, ToTensor

        def signal(eeg, baseline=None, **kwargs):
            eeg = BaselineRemoval(baseline)(eeg)
            return {'eeg': ToTensor()(eeg), 'baseline': ToTensor()(baseline),
                    'quality': 0.5}

        def label(info, **kwargs):
            return {'label': info['label'], 'weight': 1.0}

        dataset = IODataset(MemoryEEGIO(), MemoryInfoIO(),
                            transform=signal, label_transform=label)
        dataset.extend({'eeg': np.full((2, 4), 3), 'baseline': np.ones((2, 4)),
                        'info': {'label': i}} for i in range(2))
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        self.assertEqual(
            set(batch), {'eeg', 'baseline', 'label', 'quality', 'weight'})
        np.testing.assert_array_equal(
            batch['eeg'].numpy(), np.full((2, 2, 4), 2))
        np.testing.assert_array_equal(
            batch['baseline'].numpy(), np.ones((2, 2, 4)))
        self.assertEqual(batch['label'].tolist(), [0, 1])
        raw = IODataset(dataset.eeg_io, dataset.info_io)
        np.testing.assert_array_equal(raw[0]['eeg'], np.full((2, 4), 3))
        np.testing.assert_array_equal(raw[0]['baseline'], np.ones((2, 4)))
        dropped = IODataset(dataset.eeg_io, dataset.info_io,
                            transform=drop_baseline, label_transform=label)
        self.assertNotIn('baseline', next(
            iter(DataLoader(dropped, batch_size=2))))

    def test_transform_results_require_their_named_field(self):
        raw = self.make_dataset()
        for name, key in (('transform', 'eeg'), ('label_transform', 'label')):
            for result, error in (([], TypeError), ({}, ValueError), ({key: None}, ValueError)):
                with self.subTest(name=name, result=result), self.assertRaises(error):
                    IODataset(raw.eeg_io, raw.info_io,
                              **{name: lambda value, **kwargs: result})[0]
            with self.assertRaises(TypeError):
                IODataset(raw.eeg_io, raw.info_io, **{name: 1})
