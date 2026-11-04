import unittest

from torcheeg import transforms
from torcheeg.transforms import Select


class TestInputValidation(unittest.TestCase):

    def test_label_input_type_is_checked(self):
        with self.assertRaises(TypeError):
            transforms.Select('label')([1, 2])


class TestSelectTransforms(unittest.TestCase):

    def test_select(self):
        info = {'valence': 4.5, 'arousal': 5.5, 'subject_id': 7}
        transformed_label = Select(key='valence')(info)
        self.assertEqual(transformed_label, 4.5)
        transformed_label = Select(key=['valence', 'arousal'])(info)
        self.assertEqual(transformed_label, [4.5, 5.5])


if __name__ == '__main__':
    unittest.main()
