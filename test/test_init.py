import logging
import unittest
from unittest.mock import patch

import torcheeg


class TestLogging(unittest.TestCase):
    def setUp(self):
        self._logger = logging.getLogger('torcheeg')
        handlers, level = self._logger.handlers[:], self._logger.level
        self.addCleanup(self._restore_logger, handlers, level)

    def _restore_logger(self, handlers, level):
        for handler in self._logger.handlers:
            if handler not in handlers:
                handler.close()
        self._logger.handlers[:] = handlers
        self._logger.setLevel(level)

    def test_log_level_is_case_insensitive(self):
        with patch('torcheeg.mne.set_log_level') as set_level, patch(
                'torcheeg.logging.getLogger', return_value=self._logger):
            torcheeg.set_log_level('debug', third_party='ERROR')
        self.assertEqual(self._logger.level, logging.DEBUG)
        set_level.assert_called_once_with('ERROR')
        self.assertIsInstance(self._logger.handlers[-1], logging.StreamHandler)

    def test_invalid_log_level_does_not_change_configuration(self):
        with patch('torcheeg.mne.set_log_level') as set_level:
            with self.assertRaisesRegex(ValueError, 'Invalid level'):
                torcheeg.set_log_level('unknown')
        set_level.assert_not_called()


if __name__ == '__main__':
    unittest.main()
