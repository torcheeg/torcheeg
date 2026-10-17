import unittest

from torcheeg.model_selection import TrainValTestSplitGroupbyFolder

from ._three_way_cases import ThreeWayChecks, make_three_way_dataset


class TestTrainValTestSplitGroupbyFolder(ThreeWayChecks, unittest.TestCase):
    splitter_class = TrainValTestSplitGroupbyFolder
    parameters = {}
    fold_count = 1
    group_column = None
    metadata_column = 'split'

    def test_train_val_test_split_groupby_folder(self):
        dataset = make_three_way_dataset()
        train, val, test = next(self.make_splitter().split(dataset))
        self.assertEqual([len(train), len(val), len(test)], [42, 42, 41])
        for subset, name in zip((train, val, test), ('train', 'val', 'test')):
            self.assertEqual(set(subset.info['split']), {name})

    def test_unknown_folder_labels_are_rejected(self):
        dataset = make_three_way_dataset()
        dataset.info.loc[0, 'split'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'split column'):
            list(self.make_splitter().split(dataset))

    def test_empty_subset_is_rejected(self):
        dataset = make_three_way_dataset()
        dataset.info['split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            list(self.make_splitter().split(dataset))


if __name__ == '__main__':
    unittest.main()
