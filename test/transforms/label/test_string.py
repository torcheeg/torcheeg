import unittest

from torcheeg.transforms import StringToInt


class TestStringTransforms(unittest.TestCase):

    def test_string_to_int(self):
        transform = StringToInt()
        self.assertEqual(transform('None'), 0)
        self.assertEqual(transform('sub001'), 1)
        self.assertEqual(transform(['None', 'sub001']), [0, 1])
        self.assertEqual(transform(['sub001', '4']), [1, 4])


if __name__ == '__main__':
    unittest.main()
