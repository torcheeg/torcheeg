import unittest

from torcheeg.transforms import BinariesToCategory, Binary, BinaryOneVSRest


class TestBinaryTransforms(unittest.TestCase):

    def test_binary(self):
        self.assertEqual(Binary(threshold=5.0)(y=4.5)['y'], 0)
        self.assertEqual(Binary(threshold=5.0)(y=5.5)['y'], 1)
        self.assertEqual(Binary(threshold=5.0)(y=[4.5, 5.5])['y'], [0, 1])

    def test_binaries_to_multiple(self):
        self.assertEqual(BinariesToCategory()(y=[0, 0])['y'], 0)
        self.assertEqual(BinariesToCategory()(y=[0, 1])['y'], 1)
        self.assertEqual(BinariesToCategory()(y=[1, 0])['y'], 2)
        self.assertEqual(BinariesToCategory()(y=[1, 1])['y'], 3)

    def test_binary_one_vs_rest(self):
        self.assertEqual(BinaryOneVSRest(positive=1)(y=1)['y'], 1)
        self.assertEqual(BinaryOneVSRest(positive=1)(y=2)['y'], 0)
        self.assertEqual(BinaryOneVSRest(positive=1)(y=[1, 2])['y'], [1, 0])


if __name__ == '__main__':
    unittest.main()
