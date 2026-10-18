from copy import copy
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from torcheeg.datasets import SequenceDataset
from torcheeg.model_selection import train_test_split


class EpochDataset(Dataset):
    def __init__(self, info=None):
        if info is None:
            info = pd.DataFrame([dict(subject_id='person', record_id='night', clip_id=f'epoch_{i}',
                                      start_at=i * 30., end_at=(i + 1) * 30., value=i)
                                 for i in range(7)])
        self.info = info.copy()

    def __getitem__(self, index):
        value = self.info.iloc[index]['value']
        return np.full((2, 4), value, dtype=np.float32), int(value) % 5

    def __len__(self):
        return len(self.info)


class StructuredEpochDataset(EpochDataset):
    def __getitem__(self, index):
        signal, label = super().__getitem__(index)
        return signal.astype(np.float64), {'stage': label, 'clip': self.info.iloc[index]['clip_id']}


class MalformedEpochDataset(EpochDataset):
    def __getitem__(self, index):
        signal, label = super().__getitem__(index)
        return signal, label, 0


class TestSequenceDataset(unittest.TestCase):
    def test_nonoverlap_tail_labels_provenance_and_data_loader(self):
        source = EpochDataset()
        sequences = SequenceDataset(source, sequence_length=3, domain_id=2)
        self.assertEqual(len(sequences), 2)
        self.assertEqual(sequences.get_epoch_indices(1), (3, 4, 5))
        x, y, domain = sequences[1]
        torch.testing.assert_close(x[:, 0, 0], torch.tensor([3., 4., 5.]))
        torch.testing.assert_close(y, torch.tensor([3, 4, 0]))
        self.assertEqual(domain, 2)
        batches = next(iter(DataLoader(sequences, batch_size=2)))
        self.assertEqual(batches[0].shape, (2, 3, 2, 4))
        torch.testing.assert_close(batches[2], torch.tensor([2, 2]))
        self.assertEqual(sequences.read_info(1)['start_at'], 90.)
        self.assertEqual(sequences.read_info(1)['end_at'], 180.)
        self.assertEqual(len(sequences.get_labels()), 2)

    def test_unsorted_rows_nondefault_index_and_overlapping_stride(self):
        source = EpochDataset()
        source.info = source.info.iloc[[6, 1, 3, 0, 2, 5, 4]]
        source.info.index = [20, 20, 5, -1, 2, 17, 9]
        sequences = SequenceDataset(source, sequence_length=3, stride=2)
        self.assertEqual(len(sequences), 3)
        self.assertEqual(sequences.get_epoch_indices(0), (3, 1, 4))
        torch.testing.assert_close(sequences[2][0][:, 0, 0], torch.tensor([4., 5., 6.]))
        self.assertEqual(len(sequences[0]), 2)

    def test_gaps_start_new_runs_instead_of_filling_or_joining(self):
        source = EpochDataset()
        source.info.loc[3:, ['start_at', 'end_at']] += 90
        sequences = SequenceDataset(source, sequence_length=3, stride=1)
        self.assertEqual([sequences.get_epoch_indices(i) for i in range(len(sequences))],
                         [(0, 1, 2), (3, 4, 5), (4, 5, 6)])
        source.info = source.info.iloc[[0, 1, 3, 4]]
        self.assertEqual(len(SequenceDataset(source, sequence_length=3)), 0)

    def test_all_builtin_boundaries_remain_when_extra_groups_are_requested(self):
        for column in ('dataset_source', 'subject_id', 'session_id', 'record_id', 'trial_id', '_record_id'):
            source = EpochDataset()
            source.info[column] = ['first'] * 3 + ['second'] * 4
            source.info['partition'] = 'train'
            sequences = SequenceDataset(source, sequence_length=3, groupby=['partition'])
            self.assertEqual([sequences.get_epoch_indices(i) for i in range(len(sequences))],
                             [(0, 1, 2), (3, 4, 5)], column)

    def test_partition_column_and_source_domains_are_sequence_boundaries(self):
        source = EpochDataset()
        source.info['partition'] = ['train'] * 3 + ['test'] * 4
        source.info['domain'] = [0] * 3 + [1] * 4
        sequences = SequenceDataset(source, sequence_length=3, groupby=['partition'], domain_column='domain')
        self.assertEqual([sequences[i][2] for i in range(len(sequences))], [0, 1])
        self.assertEqual(sequences.info['partition'].tolist(), ['train', 'test'])
        source.info['partition'] = ['train', 'test', 'train', 'test', 'train', 'test', 'train']
        self.assertEqual(len(SequenceDataset(source, sequence_length=2, groupby=['partition'])), 0)

    def test_duplicate_epoch_identity_cannot_be_assigned_to_two_partitions(self):
        source = EpochDataset()
        source.info.loc[1, 'clip_id'] = source.info.loc[0, 'clip_id']
        source.info['partition'] = ['train', 'test', 'train', 'train', 'train', 'train', 'train']
        with self.assertRaisesRegex(ValueError, 'same source epoch'):
            SequenceDataset(source, sequence_length=2, groupby=['partition'])

    def test_domain_code_comes_from_the_bound_source_after_csv_metadata_edits(self):
        source = EpochDataset()
        source.info['domain'] = 2
        sequences = SequenceDataset(source, sequence_length=3, domain_column='domain')
        sequences.info['domain'] = 8
        self.assertEqual(sequences[0][2], 2)

    def test_duplicate_and_overlapping_epochs_raise(self):
        for start in (0., 15., 29.):
            source = EpochDataset()
            source.info.loc[1, ['start_at', 'end_at']] = [start, start + 30]
            with self.subTest(start=start), self.assertRaisesRegex(ValueError, 'overlapping'):
                SequenceDataset(source, sequence_length=2)

    def test_metadata_snapshot_does_not_mutate_or_follow_the_callers_frame(self):
        source = EpochDataset()
        original = source.info.copy(deep=True)
        sequences = SequenceDataset(source, sequence_length=3)
        pd.testing.assert_frame_equal(source.info, original)
        source.info.loc[0, 'value'] = 100
        source.info = source.info.iloc[::-1]
        torch.testing.assert_close(sequences[0][0][:, 0, 0], torch.tensor([0., 1., 2.]))
        self.assertEqual(sequences.dataset.info.index.tolist(), original.index.tolist())

    def test_csv_split_and_shallow_copy_keep_the_correct_windows(self):
        sequences = SequenceDataset(EpochDataset(), sequence_length=2)
        with tempfile.TemporaryDirectory() as root:
            train, test = train_test_split(sequences, test_size=1, split_path=root + '/split')
            again, other = train_test_split(sequences, test_size=1, split_path=root + '/split')
            self.assertEqual(len(train), 2)
            self.assertEqual(len(test), 1)
            torch.testing.assert_close(test[0][0], sequences[2][0])
            torch.testing.assert_close(other[0][1], test[0][1])
            torch.testing.assert_close(again[1][0], train[1][0])
        copied = copy(sequences)
        copied.info = sequences.info.iloc[[2, 0]].copy()
        self.assertEqual(copied.get_epoch_indices(0), (4, 5))
        self.assertEqual(copied.get_epoch_indices(1), (0, 1))

    def test_stale_sequence_rows_fail_for_reordered_source_or_changed_configuration(self):
        source = EpochDataset()
        original = SequenceDataset(source, sequence_length=3)
        changes = [SequenceDataset(source, sequence_length=2),
                   SequenceDataset(source, sequence_length=3, stride=1),
                   SequenceDataset(source, sequence_length=3, domain_id=2)]
        source.info = source.info.iloc[::-1]
        changes.append(SequenceDataset(source, sequence_length=3))
        for changed in changes:
            changed.info = original.info.copy()
            with self.assertRaisesRegex(ValueError, 'new split_path'):
                changed[0]

    def test_ids_are_value_stable_and_preserve_identifier_types(self):
        source = EpochDataset()
        first = SequenceDataset(source, sequence_length=3)
        source.info['clip_id'] = source.info['clip_id'].map(lambda x: x.encode().decode())
        second = SequenceDataset(source, sequence_length=3)
        self.assertEqual(first.info['sequence_id'].tolist(), second.info['sequence_id'].tolist())
        source.info['subject_id'] = 1
        numeric = SequenceDataset(source, sequence_length=3)
        source.info['subject_id'] = '1'
        string = SequenceDataset(source, sequence_length=3)
        self.assertNotEqual(numeric.info['sequence_id'].tolist(), string.info['sequence_id'].tolist())

    def test_pickle_and_worker_loader_preserve_order(self):
        source = SequenceDataset(EpochDataset(), sequence_length=3, domain_id=0)
        restored = pickle.loads(pickle.dumps(source))
        torch.testing.assert_close(restored[1][0], source[1][0])
        batches = list(DataLoader(restored, batch_size=2, num_workers=2, multiprocessing_context='spawn'))
        self.assertEqual(len(batches), 1)
        torch.testing.assert_close(batches[0][0][1], source[1][0])

    def test_invalid_configuration_and_domain_codes_raise(self):
        for kwargs in [dict(sequence_length=0), dict(sequence_length=True), dict(stride=1.5),
                       dict(epoch_duration=0), dict(epoch_duration=np.nan), dict(domain_id=-1),
                       dict(domain_id=True), dict(domain_id=1, domain_column='domain'),
                       dict(groupby='record_id'), dict(groupby=['record_id', 'record_id']),
                       dict(groupby=['start_at']), dict(domain_column=''), dict(domain_column=[])]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SequenceDataset(EpochDataset(), **kwargs)
        for value in (True, -1, 1.5, np.nan, 2**64):
            source = EpochDataset()
            source.info['domain'] = value
            with self.subTest(domain=value), self.assertRaises(ValueError):
                SequenceDataset(source, domain_column='domain')

    def test_invalid_or_missing_source_metadata_is_rejected(self):
        for column in ('clip_id', 'start_at', 'end_at', 'record_id'):
            source = EpochDataset()
            source.info = source.info.drop(columns=[column])
            with self.subTest(missing=column), self.assertRaises(ValueError):
                SequenceDataset(source)
        for column, value in [('record_id', None), ('start_at', np.inf), ('start_at', '0'),
                              ('start_at', -1), ('end_at', 20), ('end_at', np.nan)]:
            source = EpochDataset()
            source.info[column] = source.info[column].astype(object)
            source.info.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                SequenceDataset(source)
        with self.assertRaises(TypeError):
            SequenceDataset(object())
        source = EpochDataset()
        source.info = pd.concat([source.info, source.info[['clip_id']]], axis=1)
        with self.assertRaises(ValueError):
            SequenceDataset(source)

    def test_numeric_signal_dtype_and_structured_labels_are_preserved(self):
        sequences = SequenceDataset(StructuredEpochDataset(), sequence_length=3)
        signal, labels = sequences[0]
        self.assertEqual(signal.dtype, torch.float64)
        torch.testing.assert_close(labels['stage'], torch.tensor([0, 1, 2]))
        self.assertEqual(labels['clip'], ['epoch_0', 'epoch_1', 'epoch_2'])
        with self.assertRaisesRegex(ValueError, 'exactly'):
            SequenceDataset(MalformedEpochDataset(), sequence_length=3)[0]

    def test_explicit_epoch_duration_and_positive_real_intervals(self):
        source = EpochDataset()
        source.info[['start_at', 'end_at']] /= 30.
        sequences = SequenceDataset(source, sequence_length=3, epoch_duration=1.)
        self.assertEqual(sequences.read_info(1)['start_at'], 3.)
        self.assertEqual(sequences.read_info(1)['end_at'], 6.)
        source.info[['start_at', 'end_at']] = 0.
        with self.assertRaisesRegex(ValueError, 'epoch_duration'):
            SequenceDataset(source, epoch_duration=1e-10)

    def test_empty_input_and_explicit_custom_record_columns(self):
        source = EpochDataset()
        source.info = source.info.rename(columns={'record_id': 'night'})
        self.assertEqual(len(SequenceDataset(source, sequence_length=3, groupby=['night'])), 2)
        source.info = source.info.iloc[:0]
        empty = SequenceDataset(source, groupby=['night'])
        self.assertEqual(len(empty), 0)
        self.assertEqual(empty.get_labels(), [])


if __name__ == '__main__':
    unittest.main()
