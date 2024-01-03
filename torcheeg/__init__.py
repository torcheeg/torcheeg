import mne
import logging

__version__ = '1.1.1'


def set_log_level(level: str = "INFO", third_party: str = "CRITICAL"):
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

    mne.set_log_level(third_party)
    logging.getLogger("lightning.pytorch").setLevel(third_party)
    logging.getLogger("matplotlib").setLevel(third_party)

    log = logging.getLogger('torcheeg')
    log.setLevel(level)
    
    handler = logging.StreamHandler()

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s (%(name)s/%(threadName)s) %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    log.addHandler(handler)


set_log_level()