import atexit
import logging
import os
import random
import shutil
import ssl
import string
import sys
import tempfile
import time
import urllib
from functools import partial
from typing import Optional

log = logging.getLogger('torcheeg')


def get_temp_dir_path():
    temp_dir_path = tempfile.mkdtemp()
    atexit.register(partial(shutil.rmtree, temp_dir_path, ignore_errors=True))
    return temp_dir_path


def get_random_dir_path(root_path='.torcheeg', dir_prefix='tmp'):
    """
    Generates a unique folder name based on the current timestamp and a random suffix.
    The folder name is intended to be used for creating a temporary cache folder in torcheeg.

    Returns:
        str: A unique folder name.
    """
    timestamp = int(time.time() * 1000)

    random_suffix = ''.join(
        random.choices(string.ascii_letters + string.digits, k=5))

    dir_path = f"{dir_prefix}_{timestamp}_{random_suffix}"

    return os.path.join(root_path, dir_path)


def get_package_dir_path():
    home_dir = None

    if 'nt' == os.name.lower():
        if os.path.isdir(os.path.join(os.getenv('APPDATA'), '.torcheeg')):
            home_dir = os.getenv('APPDATA')
        else:
            home_dir = os.getenv('USERPROFILE')
    else:
        home_dir = os.path.expanduser('~')

    if home_dir is None:
        raise RuntimeError(
            'Cannot resolve home dictionary, please report this bug to TorchEEG Teams.'
        )

    return os.path.join(home_dir, '.torcheeg')


def download_url(url: str,
                 folder: str,
                 verbose: bool = True,
                 filename: Optional[str] = None):
    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = os.path.join(folder, filename)

    if os.path.exists(path):  # pragma: no cover
        if verbose:
            log.info(f'Using existing file {filename}', file=sys.stderr)
        return path

    if verbose:
        log.info(f'Downloading {url}', file=sys.stderr)

    os.makedirs(os.path.expanduser(os.path.normpath(folder)))

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path