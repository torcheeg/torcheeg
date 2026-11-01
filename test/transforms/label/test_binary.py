import unittest

from torcheeg.transforms import BinariesToCategory, Binary, BinaryOneVSRest


class TestBinaryTransforms(unittest.TestCase):

    def test_binary(self):
        self.assertEqual(Binary(threshold=5.0)(4.5), 0)
        self.assertEqual(Binary(threshold=5.0)(5.5), 1)
        self.assertEqual(Binary(threshold=5.0)([4.5, 5.5]), [0, 1])

    def test_binaries_to_multiple(self):
        self.assertEqual(BinariesToCategory()([0, 0]), 0)
        self.assertEqual(BinariesToCategory()([0, 1]), 1)
        self.assertEqual(BinariesToCategory()([1, 0]), 2)
        self.assertEqual(BinariesToCategory()([1, 1]), 3)

    def test_binary_one_vs_rest(self):
        self.assertEqual(BinaryOneVSRest(positive=1)(1), 1)
        self.assertEqual(BinaryOneVSRest(positive=1)(2), 0)
        self.assertEqual(BinaryOneVSRest(positive=1)([1, 2]), [1, 0])


if __name__ == '__main__':
    unittest.main()
