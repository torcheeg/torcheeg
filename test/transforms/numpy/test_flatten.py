import unittest

import numpy as np

from torcheeg.transforms import Flatten


class TestFlattenTransforms(unittest.TestCase):

    def test_flatten(self):
        eeg = np.random.randn(62, 5)
        transformed_eeg = Flatten()(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (310,))


if __name__ == '__main__':
    unittest.main()
