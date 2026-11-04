import unittest

import numpy as np

from torcheeg.datasets import DEAPDataset
from torcheeg.transforms import OrderElectrode, PickElectrode


class TestPickTransforms(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_order_electrode(self):
        eeg = np.random.rand(6, 3000)
        source_electrodes = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']
        target_electrodes = ['F3', 'F4', 'C3']
        transformed_eeg = OrderElectrode(
            source_electrodes=source_electrodes, target_electrodes=target_electrodes)(eeg)
        self.assertEqual(transformed_eeg.shape, (3, 3000))
        np.testing.assert_array_equal(transformed_eeg, eeg[:3])

    def test_pick_electrode(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = PickElectrode([1, 2])(eeg)
        self.assertEqual(transformed_eeg.shape, (2, 128))
        np.testing.assert_array_equal(transformed_eeg, eeg[[1, 2]])
        pick = ['FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                'O1', 'FP2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
        pick_list = PickElectrode.to_index_list(pick, DEAPDataset.channels)
        self.assertEqual(len(pick_list), 28)
        transformed_eeg = PickElectrode(
            PickElectrode.to_index_list(pick, DEAPDataset.channels))(eeg)
        self.assertEqual(transformed_eeg.shape, (28, 128))
        np.testing.assert_array_equal(transformed_eeg, eeg[pick_list])


if __name__ == '__main__':
    unittest.main()
