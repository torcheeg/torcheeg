import os
import tempfile
import unittest

from torcheeg.model_selection import KFoldPerSubject

from .test_utils import _make_dataset


class TestKFoldPerSubject(unittest.TestCase):

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self._path = os.path.join(directory.name, 'split')

    def test_each_subject_has_disjoint_complete_folds(self):
        dataset = _make_dataset()
        original = dataset.info.copy(deep=True)
        splitter = KFoldPerSubject(n_splits=5, split_path=self._path)
        folds = list(splitter.split(dataset))
        self.assertEqual(len(folds), 25)
        for subject in range(5):
            seen = []
            selected = list(splitter.split(dataset, subject=subject))
            self.assertEqual(len(selected), 5)
            expected = set(
                dataset.info.loc[dataset.info.subject_id.eq(subject), 'clip_id'])
            for train, test in selected:
                self.assertEqual(set(train.info.subject_id), {subject})
                self.assertEqual(set(test.info.subject_id), {subject})
                a, b = (set(train.info.clip_id), set(test.info.clip_id))
                self.assertFalse(a & b)
                self.assertEqual(a | b, expected)
                seen.extend(test.info.clip_id)
            self.assertEqual(set(seen), expected)
            self.assertEqual(len(seen), len(expected))
        self.assertTrue(original.equals(dataset.info))
        self.assertEqual(splitter.fold_ids, list(range(5)))

    def test_seeded_splits_are_reproducible_across_independent_caches(self):
        dataset = _make_dataset()
        selections = []
        for suffix in ('a', 'b'):
            splitter = KFoldPerSubject(
                n_splits=5, shuffle=True, random_state=7, split_path=self._path + suffix)
            selections.append([test.info.clip_id.tolist()
                              for _, test in splitter.split(dataset)])
        self.assertEqual(selections[0], selections[1])

    def test_invalid_fold_count_is_rejected(self):
        with self.assertRaises(ValueError):
            KFoldPerSubject(n_splits=1, split_path=self._path)


if __name__ == '__main__':
    unittest.main()
