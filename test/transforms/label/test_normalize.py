import unittest

from torcheeg.transforms import Normalize


class TestNormalizeTransforms(unittest.TestCase):

    def test_normalize(self):
        t = Normalize(min=0.0, max=1.0)
        self.assertEqual(t(0.5), 0.5)


if __name__ == '__main__':
    unittest.main()
