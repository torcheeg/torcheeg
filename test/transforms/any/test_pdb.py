import importlib
import unittest
from unittest.mock import patch

import numpy as np

from torcheeg.transforms import PDB


class TestPDB(unittest.TestCase):
    def test_breakpoint_preserves_selected_and_unselected_inputs(self):
        eeg, baseline = np.arange(4), np.arange(2)
        for targets in (['eeg'], ['eeg', 'baseline', 'y']):
            with self.subTest(targets=targets), patch.object(
                    importlib.import_module('torcheeg.transforms.any.pdb').pdb,
                    'set_trace') as breakpoint:
                result = PDB(targets)(eeg=eeg, baseline=baseline, y='rest')
            breakpoint.assert_called_once_with()
            self.assertIs(result['eeg'], eeg)
            self.assertIs(result['baseline'], baseline)
            self.assertEqual(result['y'], 'rest')


if __name__ == '__main__':
    unittest.main()
