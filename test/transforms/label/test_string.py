import unittest

from torcheeg.transforms import StringToInt


class TestStringTransforms(unittest.TestCase):

    def test_string_to_int(self):
        transform = StringToInt()

        self.assertEqual(transform(y='None')['y'], 0)
        self.assertEqual(transform(y='sub001')['y'], 1)

        self.assertEqual(transform(y=['None', 'sub001'])['y'], [0, 1])
        self.assertEqual(transform(y=['sub001', '4'])['y'], [1, 4])


if __name__ == '__main__':
    unittest.main()
