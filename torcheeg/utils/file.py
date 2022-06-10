# from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/download.py

import os
import ssl
import sys
import urllib
from typing import Optional

import errno


def makedirs(path: str):
    r"""Recursive directory creation function."""
    try:
        os.makedirs(os.path.expanduser(os.path.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise e


def download_url(url: str, folder: str, verbose: bool = True, filename: Optional[str] = None):
    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = os.path.join(folder, filename)

    if os.path.exists(path):  # pragma: no cover
        if verbose:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if verbose:
        print(f'Downloading {url}', file=sys.stderr)

    makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path