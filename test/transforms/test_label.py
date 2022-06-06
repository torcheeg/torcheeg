import unittest

from torcheeg.transforms import Select, Binary, BinariesToCategory


class TestPyGTransforms(unittest.TestCase):
    def test_select(self):
        label = {'valence': 4.5, 'arousal': 5.5, 'subject': 7}
        transformed_label = Select(key='valence')(label)
        self.assertEqual(transformed_label, 4.5)

        transformed_label = Select(key=['valence', 'arousal'])(label)
        self.assertEqual(transformed_label, [4.5, 5.5])

    def test_binary(self):
        self.assertEqual(Binary(threshold=5.0)(4.5), 0)
        self.assertEqual(Binary(threshold=5.0)(5.5), 1)
        self.assertEqual(Binary(threshold=5.0)([4.5, 5.5]), [0, 1])

    def test_binaries_to_multiple(self):
        self.assertEqual(BinariesToCategory()([0, 0]), 0)
        self.assertEqual(BinariesToCategory()([0, 1]), 1)
        self.assertEqual(BinariesToCategory()([1, 0]), 2)
        self.assertEqual(BinariesToCategory()([1, 1]), 3)


if __name__ == '__main__':
    unittest.main()
