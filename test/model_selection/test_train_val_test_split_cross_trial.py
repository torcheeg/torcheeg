import tempfile
import unittest

from torcheeg.model_selection import TrainValTestSplitCrossTrial

from ._three_way_cases import ThreeWayChecks, make_three_way_dataset


class TestTrainValTestSplitCrossTrial(ThreeWayChecks, unittest.TestCase):
    splitter_class = TrainValTestSplitCrossTrial
    parameters = {'train_trials': [0, 1, 2], 'val_trials': [3], 'test_trials': [4]}
    fold_count = 1
    group_column = 'trial_id'
    metadata_column = 'trial_id'

    def test_train_val_test_split_cross_trial(self):
        train, val, test = next(self.make_splitter().split(make_three_way_dataset()))
        self.assertEqual([len(train), len(val), len(test)], [75, 25, 25])

    def test_overlapping_empty_duplicate_and_missing_partition_ids_are_rejected(self):
        for ids in ([0, 1, 3], [], [0, 0, 1], [None]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.make_splitter(train_trials=ids)

    def test_unknown_partition_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            list(self.make_splitter(train_trials=[0, 1, 99]).split(make_three_way_dataset()))

    def test_unassigned_ids_can_be_intentionally_excluded(self):
        train, val, test = next(self.make_splitter(train_trials=[0],
            val_trials=[1], test_trials=[2]).split(make_three_way_dataset()))
        self.assertEqual([len(train), len(val), len(test)], [25, 25, 25])

    def test_changed_partition_configuration_does_not_reuse_old_cache(self):
        dataset = make_three_way_dataset()
        list(self.make_splitter().split(dataset))
        with self.assertRaisesRegex(ValueError, 'parameters'):
            next(self.make_splitter(train_trials=[4], test_trials=[0, 1, 2]).split(dataset))

    def test_string_ids_keep_their_values_and_types(self):
        dataset = make_three_way_dataset()
        dataset.info['trial_id'] = dataset.info['trial_id'].map(lambda value: f'id_{value:02d}')
        splitter = self.make_splitter(train_trials=['id_00', 'id_01', 'id_02'],
            val_trials=['id_03'], test_trials=['id_04'])
        folds = list(splitter.split(dataset))
        self.assert_partition(dataset, folds)
        self.assertEqual(set(folds[0][1].info['trial_id']), {'id_03'})

    def test_caller_id_lists_are_copied(self):
        ids = [0, 1, 2]
        splitter = self.make_splitter(train_trials=ids)
        ids.append(3)
        self.assertEqual(splitter.train_trials, [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
