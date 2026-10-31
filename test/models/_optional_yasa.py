"""Run complete optional reference checks only with the supported backend."""

import sys


def matching_yasa_available():
    if sys.version_info < (3, 11):
        return False
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version('yasa') == '0.8.1'
    except PackageNotFoundError:
        return False
