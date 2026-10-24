import os
import tempfile
import unittest

from torcheeg.model_selection import Subcategory

from .test_utils import _make_dataset


class TestSubcategory(unittest.TestCase):

    def test_string_numeric_and_mixed_categories_preserve_values(self):
        for categories in ([0, 1, 2, 3, 4], ['001', '002', '003', '004', '005'],
                           ['group/a', 'group%b', 'rest', 'task',
                            'other'], [0, '0', 2, '2', 4]):
            with self.subTest(
                    categories=categories), tempfile.TemporaryDirectory() as path:
                dataset = _make_dataset()
                dataset.info['task'] = dataset.info.subject_id.map(
                    dict(enumerate(categories)))
                splitter = Subcategory(split_path=os.path.join(path, 'split'))
                first = list(splitter.split(dataset))
                cached = list(
                    Subcategory(split_path=splitter.split_path).split(dataset))
                self.assertEqual(len(first), 5)
                self.assertEqual(set(splitter.category_list), set(categories))
                for subset, repeated in zip(first, cached):
                    self.assertEqual(subset.info.task.nunique(), 1)
                    self.assertEqual(subset.info.clip_id.tolist(),
                                     repeated.info.clip_id.tolist())
                self.assertEqual(
                    set().union(*(set(subset.info.clip_id) for subset in first)),
                    set(dataset.info.clip_id))


if __name__ == '__main__':
    unittest.main()
