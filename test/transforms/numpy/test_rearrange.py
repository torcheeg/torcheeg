import unittest

import numpy as np

from torcheeg.transforms import RearrangeElectrode


class TestRearrangeTransforms(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_rearrange_electrode(self):
        src_eeg = np.random.rand(3, 128)
        tgt_eeg = RearrangeElectrode(source=['FP1', 'F3', 'F7'],
                                     target=['F3', 'F7', 'FP1', 'AF2'],
                                     missing='mean')(eeg=src_eeg)['eeg']
        self.assertEqual(tgt_eeg.shape, (4, 128))

        self.assertEqual(src_eeg[0, :].tolist(), tgt_eeg[2, :].tolist())
        self.assertEqual(src_eeg[1, :].tolist(), tgt_eeg[0, :].tolist())
        self.assertEqual(src_eeg[2, :].tolist(), tgt_eeg[1, :].tolist())


if __name__ == '__main__':
    unittest.main()
