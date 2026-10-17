"""Owned, resumable disk-cache generations for BaseDataset."""

import ast
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
import textwrap
import uuid

import cloudpickle
from filelock import FileLock
from joblib import hash as joblib_hash
import pandas as pd

from torcheeg.io import EEGSignalIO

log = logging.getLogger('torcheeg')
_CACHE_FILE = '.torcheeg-cache.json'
_RECORD_FILE = '.torcheeg-record.json'
_STAGE_FILE = '.torcheeg-stage.json'
_SCHEMA = 1


def _write_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.tmp-', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path):
    try:
        with open(path) as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError('Expected a manifest object.')
        return value
    except (OSError, ValueError) as error:
        raise ValueError(f'Invalid cache manifest at {path}; preserve it and use a new io_path.') from error


def _hash(value):
    try:
        return joblib_hash(value, hash_name='sha1')
    except Exception as error:
        # Dynamic callbacks, including notebook lambdas, may need cloudpickle.
        try:
            return hashlib.sha256(cloudpickle.dumps(value)).hexdigest()
        except Exception:
            raise ValueError('Cannot identify the cache configuration; use serializable preprocessing settings.') from error


def _code(function):
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.body and isinstance(node.body[0], ast.Expr):
                    value = node.body[0].value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        node.body.pop(0)
        return ast.dump(tree, include_attributes=False)
    except (OSError, TypeError, IndentationError, SyntaxError):
        return (getattr(function, '__module__', None),
                getattr(function, '__qualname__', type(function).__qualname__))


def configuration_signature(dataset, kwargs):
    parameters = {key: value for key, value in kwargs.items()
                  if key not in ('online_transform', 'label_transform')}
    callbacks = [dataset.read_record, dataset.process_record, dataset.set_records]
    for name in ('offline_transform', 'before_trial'):
        value = parameters.get(name)
        if value is not None:
            callbacks.append(value if inspect.isfunction(value) else type(value).__call__)
    hooks = [dataset.after_trial, dataset.after_session, dataset.after_subject]
    callbacks.extend(value for value in hooks if value is not None)
    return _hash({'dataset': (type(dataset).__module__, type(dataset).__qualname__),
                  'mode': dataset.io_mode, 'parameters': parameters, 'hooks': hooks,
                  'code': [_code(value) for value in callbacks]})


def source_state(kwargs, excluded):
    """Track explicit source paths using names, sizes and modification times."""
    excluded = [Path(value).resolve() for value in excluded]

    def excluded_path(path):
        resolved = path.resolve()
        return any(resolved == item or item in resolved.parents for item in excluded)

    def describe(value):
        path = Path(value).resolve()
        if not path.exists():
            return None
        if path.is_file():
            stat = path.stat()
            return [str(path), stat.st_size, stat.st_mtime_ns]
        files = []
        for directory, folders, names in os.walk(path, followlinks=False):
            folders[:] = sorted(name for name in folders
                                if not excluded_path(Path(directory) / name))
            for name in sorted(names):
                item = Path(directory) / name
                if item.is_file() and not excluded_path(item):
                    stat = item.stat()
                    files.append([str(item.relative_to(path)), stat.st_size, stat.st_mtime_ns])
        return [str(path), files]

    sources = {}
    for name in ('root_path', 'mat_path', 'file_path', 'csv_path', 'download_path'):
        value = kwargs.get(name)
        if isinstance(value, (str, os.PathLike)):
            sources[name] = describe(value)
    csv_path = kwargs.get('csv_path')
    if isinstance(csv_path, (str, os.PathLike)) and Path(csv_path).is_file():
        frame = pd.read_csv(csv_path, usecols=lambda column: column == 'file_path')
        if 'file_path' in frame:
            sources['csv_files'] = [describe(value) for value in frame['file_path'].dropna()]
    for name, value in kwargs.items():
        if isinstance(value, (list, tuple)):
            filenames = [filename for item in value if hasattr(item, 'filenames')
                         for filename in item.filenames if filename is not None]
            if filenames:
                sources[name] = [describe(filename) for filename in filenames]
    return sources


def _available_sources_match(previous, current):
    if current is None:
        return True
    if isinstance(current, list):
        return (isinstance(previous, list) and len(previous) == len(current)
                and all(_available_sources_match(old, new)
                        for old, new in zip(previous, current)))
    return previous == current


def _describe_record(path, mode, io_size):
    path = Path(path)
    metadata = path / 'info.csv'
    signals = path / 'eeg'
    if (path.is_symlink() or metadata.is_symlink() or signals.is_symlink()
            or not metadata.is_file() or not signals.is_dir()):
        raise ValueError(f'Incomplete cache record {path}; preserve it and use a new io_path.')
    content = metadata.read_bytes()
    frame = pd.read_csv(metadata) if content else pd.DataFrame()
    if '_record_id' in frame.columns or (len(frame) and 'clip_id' not in frame):
        raise ValueError(f'Invalid cache metadata in {metadata}.')
    if mode == 'lmdb' and not (signals / 'data.mdb').is_file():
        raise ValueError(f'No LMDB data file in {signals}; check io_mode.')
    signal_io = EEGSignalIO(str(signals), io_size=io_size, io_mode=mode)
    try:
        keys = set(signal_io.keys())
        required = set()
        for column in ('clip_id', 'baseline_id'):
            if column in frame:
                required.update(str(value) for value in frame[column].dropna())
        if not required.issubset(keys):
            raise ValueError(f'Missing cached clip or baseline keys in {path}.')
        if mode == 'pickle':
            if any(not (signals / key).is_file() or (signals / key).stat().st_size == 0
                   for key in keys):
                raise ValueError(f'Invalid pickle signal files in {signals}.')
        return {'metadata_sha256': hashlib.sha256(content).hexdigest(),
                'rows': len(frame), 'keys': sorted(keys)}
    finally:
        signal_io = None


def _validate_record(path, mode, io_size, signature=None):
    description = _describe_record(path, mode, io_size)
    marker = Path(path) / _RECORD_FILE
    if signature is not None:
        manifest = _read_json(marker)
        expected = manifest.get('description', {})
        if (manifest.get('schema') != _SCHEMA or manifest.get('signature') != signature
                or expected.get('metadata_sha256') != description['metadata_sha256']
                or expected.get('rows') != description['rows']
                or not set(expected.get('keys', [])).issubset(description['keys'])):
            raise ValueError(f'Cache record validation failed at {path}; preserve it and use a new io_path.')
    return description


def process_cached_record(io_path, io_size, io_mode, record, record_id,
                          handle_record, signature, **kwargs):
    clean = Path(io_path)
    target = clean / f'_record_{record_id}'
    record_signature = _hash((signature, record))
    if target.exists():
        _validate_record(target, io_mode, io_size, record_signature)
        return {'record': target.name}
    temporary = Path(tempfile.mkdtemp(prefix=f'.pending-{record_id}-', dir=str(clean.parent)))
    # A failed record stays unpublished; completed sibling records remain reusable.
    result = handle_record(io_path=str(temporary), io_size=io_size,
                           io_mode=io_mode, record=record, record_id=record_id, **kwargs)
    generated = temporary / result['record']
    description = _describe_record(generated, io_mode, io_size)
    _write_json(generated / _RECORD_FILE,
                {'schema': _SCHEMA, 'signature': record_signature, 'description': description})
    os.replace(generated, target)
    shutil.rmtree(temporary)
    return result


class DatasetCache:
    def __init__(self, path, mode, io_size):
        self.path = Path(os.path.normcase(os.path.realpath(os.fspath(path))))
        self.stage = self.path.parent / ('.' + self.path.name + '.torcheeg-stage')
        self.lock_path = self.path.parent / ('.' + self.path.name + '.torcheeg-lock')
        self.mode = mode
        self.io_size = io_size
        self.clean = self.stage / 'clean'
        self.work = self.stage / 'work'

    def lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(str(self.lock_path))

    def sources(self, kwargs):
        temporary = list(self.path.parent.glob('.' + self.path.name + '.torcheeg-new-*'))
        temporary += list(self.path.parent.glob('.' + self.path.name + '.torcheeg-retired-*'))
        return source_state(kwargs, [self.path, self.stage, self.lock_path] + temporary)

    def validate_existing(self, signature, sources):
        if not self.path.exists():
            return False
        if not self.path.is_dir() or self.path.is_symlink():
            raise ValueError('io_path must be a real cache directory.')
        marker = self.path / _CACHE_FILE
        if marker.exists():
            manifest = _read_json(marker)
            if (manifest.get('schema') != _SCHEMA or manifest.get('owner') != 'torcheeg'
                    or re.fullmatch(r'[0-9a-f]{32}', str(manifest.get('token'))) is None
                    or manifest.get('state') != 'complete'
                    or manifest.get('signature') != signature):
                raise ValueError('The cache configuration differs or is incomplete; use a new io_path.')
            available = {name: value for name, value in sources.items() if value is not None}
            if any(not _available_sources_match(manifest.get('sources', {}).get(name), value)
                   for name, value in available.items()):
                raise ValueError('Source metadata changed; preserve the cache and use a new io_path.')
            records = manifest.get('records')
            if not isinstance(records, dict) or not records:
                raise ValueError('The cache record manifest is incomplete.')
            actual_records = {item.name for item in self.path.iterdir() if '_record_' in item.name}
            if actual_records != set(records):
                raise ValueError('The completed cache has unexpected or missing record paths.')
            self.complete_manifest = manifest
            for name, record_signature in records.items():
                if re.fullmatch(r'_record_[0-9]+', name) is None:
                    raise ValueError('Invalid cache record name.')
                _validate_record(self.path / name, self.mode, self.io_size, record_signature)
        else:
            records = [path for path in self.path.iterdir()
                       if path.is_dir() and re.fullmatch(r'_record_[0-9]+', path.name)]
            if not records:
                raise ValueError('Legacy cache has no records; preserve it and select a new io_path.')
            if any('_record_' in item.name and item not in records
                   for item in self.path.iterdir()):
                raise ValueError('Legacy cache has an invalid record path; preserve it for inspection.')
            for record in records:
                _validate_record(record, self.mode, self.io_size)
            log.warning('Legacy cache has no completion/configuration manifest. '
                        'Its structure is readable, but completion and preprocessing identity are unverified. '
                        'Use a new io_path if generation was interrupted or settings changed.')
        return True

    def begin(self, signature, sources, records):
        record_map = {f'_record_{i}': _hash((signature, record))
                      for i, record in enumerate(records)}
        if not record_map:
            raise ValueError('No source records were selected for the disk cache.')
        expected = {'schema': _SCHEMA, 'owner': 'torcheeg', 'signature': signature,
                    'sources': sources, 'records': record_map}
        if self.stage.exists():
            if not self.stage.is_dir() or self.stage.is_symlink():
                raise ValueError('The staging path is not an owned directory; select a new io_path.')
            manifest = _read_json(self.stage / _STAGE_FILE)
            if (any(manifest.get(key) != value for key, value in expected.items())
                    or re.fullmatch(r'[0-9a-f]{32}', str(manifest.get('token'))) is None):
                raise ValueError('The incomplete cache configuration differs; preserve it and use a new io_path.')
        else:
            temporary = Path(tempfile.mkdtemp(
                prefix='.' + self.path.name + '.torcheeg-new-', dir=str(self.path.parent)))
            manifest = dict(expected, token=uuid.uuid4().hex)
            _write_json(temporary / _STAGE_FILE, manifest)
            os.replace(temporary, self.stage)
        self.manifest = manifest
        if self.clean.is_symlink():
            raise ValueError('The clean staging path must not be a symlink.')
        self.clean.mkdir(exist_ok=True)
        # Only artifacts inside a verified owned generation are removable.
        for partial in self.stage.glob('.pending-*'):
            if partial.is_dir() and not partial.is_symlink():
                shutil.rmtree(partial)

    def hook_work(self):
        if self.work.exists():
            if self.work.is_symlink():
                raise ValueError('The hook staging path must not be a symlink.')
            shutil.rmtree(self.work)
        shutil.copytree(self.clean, self.work)
        return self.work

    def publish(self, path, sources):
        if sources != self.manifest['sources']:
            raise ValueError('Source metadata changed during preprocessing; preserve this generation.')
        path = Path(path)
        rows = 0
        for name, record_signature in self.manifest['records'].items():
            description = _describe_record(path / name, self.mode, self.io_size)
            rows += description['rows']
            _write_json(path / name / _RECORD_FILE,
                        {'schema': _SCHEMA, 'signature': record_signature, 'description': description})
        if not rows:
            raise ValueError('No samples were produced; check the preprocessing settings.')
        _write_json(path / _CACHE_FILE,
                    dict(self.manifest, state='complete'))
        if self.path.exists():
            raise ValueError('io_path appeared during generation; preserve both paths and select a new io_path.')
        os.replace(path, self.path)

    def _retirement_paths(self, token):
        retired = self.path.parent / ('.' + self.path.name + '.torcheeg-retired-' + token)
        return retired, retired.with_name(retired.name + '.owner.json')

    def _cleanup_retired(self, token):
        retired, owner = self._retirement_paths(token)
        if not owner.exists():
            return
        try:
            proof = _read_json(owner)
            if (proof.get('owner') != 'torcheeg' or proof.get('token') != token
                    or proof.get('retired') != str(retired) or retired.is_symlink()):
                raise ValueError('Retired generation ownership is invalid.')
            if retired.exists():
                shutil.rmtree(retired)
            owner.unlink()
        except (OSError, ValueError):
            log.warning('The cache is complete, but retired staging cleanup needs inspection at %s.', retired)

    def cleanup(self):
        token = self.manifest['token']
        retired, owner = self._retirement_paths(token)
        try:
            manifest = _read_json(self.stage / _STAGE_FILE)
            if manifest.get('token') != token or retired.exists():
                raise ValueError('Cache staging ownership changed.')
            # Keep proof outside the retired tree so cleanup itself can resume.
            _write_json(owner, {'owner': 'torcheeg', 'schema': _SCHEMA,
                                'token': token, 'retired': str(retired)})
            os.replace(self.stage, retired)
        except (OSError, ValueError):
            log.warning('The cache is complete; unverified staging data is preserved at %s.', self.stage)
            return
        self._cleanup_retired(token)

    def cleanup_published(self):
        if not hasattr(self, 'complete_manifest'):
            return
        token = self.complete_manifest['token']
        if self.stage.exists():
            try:
                manifest = _read_json(self.stage / _STAGE_FILE)
                if manifest.get('token') == token:
                    self.manifest = manifest
                    self.cleanup()
            except ValueError:
                log.warning('Unverified staging data is preserved at %s.', self.stage)
        self._cleanup_retired(token)
