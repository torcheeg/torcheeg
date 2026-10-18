import tempfile
import os
import unittest

from torcheeg.model_selection import TrainValTestSplitGroupbyFolder

from .test_utils import make_dataset


class TestTrainValTestSplitGroupbyFolder(unittest.TestCase):
    splitter_class = TrainValTestSplitGroupbyFolder
    parameters = {}
    fold_count = 1
    group_column = None
    metadata_column = 'split'

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.split_path = os.path.join(self.directory.name, 'split')

    def make_splitter(self, split_path=None, **overrides):
        parameters = dict(self.parameters)
        parameters.update(overrides)
        return self.splitter_class(
            split_path=self.split_path if split_path is None else split_path,
            **parameters)

    def assert_partition(self, dataset, folds):
        self.assertEqual(len(folds), self.fold_count)
        for fold in folds:
            self.assertEqual(len(fold), 3)
            clips = [set(subset.info['clip_id']) for subset in fold]
            self.assertEqual(set.union(*clips), set(dataset.info['clip_id']))
            for first, second in ((0, 1), (0, 2), (1, 2)):
                self.assertFalse(clips[first] & clips[second])
                if self.group_column is not None:
                    self.assertFalse(
                        set(fold[first].info[self.group_column])
                        & set(fold[second].info[self.group_column]))
            for subset in fold:
                self.assertGreater(len(subset), 0)
                for index in range(len(subset)):
                    signal, info = subset[index]
                    self.assertEqual(signal[0], int(info['clip_id']))

    def test_train_val_test_split_groupby_folder(self):
        dataset = make_dataset()
        train, val, test = next(self.make_splitter().split(dataset))
        self.assertEqual([len(train), len(val), len(test)], [42, 42, 41])
        for subset, name in zip((train, val, test), ('train', 'val', 'test')):
            self.assertEqual(set(subset.info['split']), {name})

    def test_unknown_folder_labels_are_rejected(self):
        dataset = make_dataset()
        dataset.info.loc[0, 'split'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'split column'):
            list(self.make_splitter().split(dataset))

    def test_empty_subset_is_rejected(self):
        dataset = make_dataset()
        dataset.info['split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            list(self.make_splitter().split(dataset))


if __name__ == '__main__':
    unittest.main()
