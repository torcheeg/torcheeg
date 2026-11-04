import unittest

from torcheeg.transforms import Mapping


class TestMappingTransforms(unittest.TestCase):

    def test_mapping(self):
        transform = Mapping({'left_hand': 0, 'right_hand': 1})
        self.assertEqual(transform('left_hand'), 0)
        self.assertEqual(transform(['left_hand', 'right_hand']), [0, 1])


if __name__ == '__main__':
    unittest.main()
