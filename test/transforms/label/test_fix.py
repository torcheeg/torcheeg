import unittest

from torcheeg.transforms import FixCategory


class TestFixTransforms(unittest.TestCase):

    def test_fix_category(self):
        self.assertEqual(FixCategory(value=0)(y=3)['y'], 0)
        self.assertEqual(FixCategory(value=[0, 1])(y=[1, 2])['y'], [0, 1])


if __name__ == '__main__':
    unittest.main()
