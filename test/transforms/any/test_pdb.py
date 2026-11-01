import unittest
from unittest.mock import patch

from torcheeg.transforms import PDB


class TestPDB(unittest.TestCase):
    def test_breakpoint_returns_input_unchanged(self):
        value = {'eeg': [1, 2], 'info': {'label': 1}}
        with patch('pdb.set_trace') as breakpoint:
            self.assertIs(PDB()(value), value)
        breakpoint.assert_called_once_with()
