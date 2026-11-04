import json
import os
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

from filelock import FileLock

from ..utils import compute_checksum


def _get_directory(eeg_io, info_io):
    if eeg_io.persistent and info_io.persistent:
        eeg_parent = os.path.dirname(os.path.abspath(eeg_io.io_path))
        info_parent = os.path.dirname(os.path.abspath(info_io.io_path))
        if eeg_parent == info_parent:
            return info_parent
    owner = info_io if info_io.persistent else eeg_io
    return os.path.abspath(owner.io_path) + '.cache'


def _collect_files(eeg_io, info_io, directory):
    files = {}
    for io in (eeg_io, info_io):
        if not io.persistent:
            continue
        path = Path(io.io_path)
        candidates = sorted(path.rglob('*')) if path.is_dir() else [path]
        for file in candidates:
            if file.is_symlink():
                raise ValueError(
                    'Invalid cache output path: symbolic links are not supported.')
            if file.is_file():
                files[os.path.relpath(file, directory)
                      ] = compute_checksum(file)
    return files


def ensure_cache(eeg_io, info_io, fingerprint, builder):
    """Build an IO pair under locks and publish its manifest last."""
    persistent = [io for io in (eeg_io, info_io) if io.persistent]
    if not persistent:
        staged = [eeg_io.clone(None), info_io.clone(None)]
        builder(*staged)
        for original, result in zip((eeg_io, info_io), staged):
            original.__dict__.update(result.__dict__)
        return
    paths = [os.path.realpath(io.io_path) for io in persistent]
    if len(set(paths)) != len(paths) or any(
            left.startswith(right + os.sep) for left in paths for right in paths if left != right):
        raise ValueError(
            'EEG and metadata paths must be separate and must not contain one another.')
    directory = _get_directory(eeg_io, info_io)
    manifest_path = os.path.join(directory, 'manifest.json')
    with ExitStack() as stack:
        for path in sorted(set(paths + [directory])):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            stack.enter_context(FileLock(path + '.lock'))
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding='utf-8') as file:
                manifest = json.load(file)
            if manifest.get('fingerprint') != fingerprint:
                raise ValueError(
                    'Cache inputs or configuration changed; use a new IO path.')
            if manifest.get('files') != _collect_files(eeg_io, info_io, directory):
                raise ValueError(
                    'Corrupt cache: output files changed or are missing.')
            if len(persistent) != 2:
                raise ValueError(
                    'Reusing a persistent cache requires both EEG and metadata to be persistent.')
            return
        for io in persistent:
            if len(io):
                raise ValueError(
                    'Incomplete cache: existing IO has no manifest; use a new IO path.')
        temporary = []
        staged = []
        published = []
        backups = []
        updates = []
        try:
            for io in (eeg_io, info_io):
                if io.persistent:
                    parent = tempfile.mkdtemp(
                        prefix='.torcheeg-cache-', dir=os.path.dirname(os.path.abspath(io.io_path)))
                    temporary.append(parent)
                    staged.append(io.clone(os.path.join(parent, 'data')))
                else:
                    staged.append(io.clone(None))
            builder(*staged)
            for original, result in zip((eeg_io, info_io), staged):
                result.close()
                if original.persistent:
                    original.close()
                    if os.path.exists(original.io_path):
                        backup = os.path.join(
                            os.path.dirname(result.io_path), 'previous')
                        os.replace(original.io_path, backup)
                        backups.append((backup, original.io_path))
                    os.replace(result.io_path, original.io_path)
                    published.append(original.io_path)
                    state = result.clone(original.io_path).__dict__
                else:
                    state = result.__dict__
                updates.append((original, state))
            manifest = {'fingerprint': fingerprint,
                        'files': _collect_files(eeg_io, info_io, directory)}
            os.makedirs(directory, exist_ok=True)
            with open(manifest_path + '.tmp', 'w', encoding='utf-8') as file:
                json.dump(manifest, file, sort_keys=True)
            os.replace(manifest_path + '.tmp', manifest_path)
            for original, state in updates:
                original.__dict__.update(state)
        except BaseException:
            for path in published:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.exists(path):
                    os.unlink(path)
            for backup, path in reversed(backups):
                os.replace(backup, path)
            if os.path.exists(manifest_path + '.tmp'):
                os.unlink(manifest_path + '.tmp')
            raise
        finally:
            for io in staged:
                io.close()
            for parent in temporary:
                shutil.rmtree(parent, ignore_errors=True)
