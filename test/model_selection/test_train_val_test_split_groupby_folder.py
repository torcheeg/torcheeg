import os
import tempfile
import unittest

from torcheeg.model_selection import TrainValTestSplitGroupbyFolder

from .test_utils import _make_dataset


class TestTrainValTestSplitGroupbyFolder(unittest.TestCase):
    _splitter_class = TrainValTestSplitGroupbyFolder
    _parameters = {}
    _fold_count = 1
    _group_column = None
    _metadata_column = 'split'

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._split_path = os.path.join(self._directory.name, 'split')

    def _make_splitter(self, split_path=None, **overrides):
        parameters = dict(self._parameters)
        parameters.update(overrides)
        return self._splitter_class(
            split_path=self._split_path if split_path is None else split_path,
            **parameters)

    def test_train_val_test_split_groupby_folder(self):
        dataset = _make_dataset()
        train, val, test = next(self._make_splitter().split(dataset))
        self.assertEqual([len(train), len(val), len(test)], [42, 42, 41])
        for subset, name in zip((train, val, test), ('train', 'val', 'test')):
            self.assertEqual(set(subset.info['split']), {name})

    def test_unknown_folder_labels_are_rejected(self):
        dataset = _make_dataset()
        dataset.info.loc[0, 'split'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'split column'):
            list(self._make_splitter().split(dataset))

    def test_empty_subset_is_rejected(self):
        dataset = _make_dataset()
        dataset.info['split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            list(self._make_splitter().split(dataset))


if __name__ == '__main__':
    unittest.main()
