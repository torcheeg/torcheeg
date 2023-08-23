__version__ = '1.1.0'

import logging

from mne import set_log_level as _set_mne_log_level


def _set_lightning_log_level(level: str = "INFO"):

    # name to level
    level = getattr(logging, level)

    # configure logging at the root level of Lightning
    logging.getLogger("lightning.pytorch").setLevel(level)


def set_log_level(level: str = "DEBUG", third_party: str = "ERROR"):
    """
    Set log level for torcheeg.

    Args:
        level (str): Log level. Choose one of ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].
        third_party (str): Log level for third party libraries. Choose one of ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].
    """
    VALID_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    level = level.upper()
    if level not in VALID_LEVELS:
        raise ValueError(
            f"Invalid level {level}. Choose one of {VALID_LEVELS}.")

    _set_mne_log_level(third_party)
    _set_lightning_log_level(third_party)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s",
    )


set_log_level()