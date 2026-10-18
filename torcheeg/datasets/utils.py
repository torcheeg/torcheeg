import functools
import inspect
import marshal
import os
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from ..utils.cache import (ensure_cache, get_cache_key as _get_cache_key,
                           validate_cache, validate_timeout, get_file_checksum)

__all__ = ['get_cache_key', 'validate_cache', 'ensure_cache', 'validate_timeout']


def _describe(value, seen=None):
    if value is None or isinstance(value, (str, bytes, int, float, bool, np.generic)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return str(tensor.dtype), tuple(tensor.shape), joblib.hash(
            tensor.reshape(-1).view(torch.uint8).numpy())
    if isinstance(value, (np.ndarray, pd.DataFrame, pd.Series)):
        return joblib.hash(value)
    if isinstance(value, types.ModuleType):
        return value.__name__
    if isinstance(value, type):
        return value.__module__, value.__qualname__
    seen = set() if seen is None else seen
    if id(value) in seen:
        return type(value).__module__, type(value).__qualname__
    seen = seen | {id(value)}
    if isinstance(value, dict):
        return {key: _describe(item, seen) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value).__name__, [_describe(item, seen) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_describe(item, seen) for item in value), key=repr)
    if isinstance(value, functools.partial):
        return (_describe(value.func,
                          seen), _describe(value.args,
                                           seen), _describe(value.keywords, seen))
    if inspect.ismethod(value):
        return (_describe(value.__func__, seen), _describe(value.__self__, seen))
    if inspect.isfunction(value):
        closure = [cell.cell_contents for cell in (value.__closure__ or ())]
        constants = {
            name: value.__globals__[name]
            for name in value.__code__.co_names
            if name in value.__globals__ and isinstance(value.__globals__[name], (
                str, bytes, int, float, bool, tuple, list, dict, np.ndarray))
        }
        return (value.__module__, value.__qualname__, marshal.dumps(value.__code__),
                _describe(value.__defaults__,
                          seen), _describe(value.__kwdefaults__, seen),
                _describe(closure, seen), _describe(constants, seen))
    if hasattr(value, '__dict__'):
        cls = type(value)
        try:
            implementation = inspect.getsource(cls)
        except (OSError, TypeError):
            implementation = cls.__module__, cls.__qualname__
        return implementation, _describe(vars(value), seen)
    return joblib.hash(value)


def _input_files(value, excluded):
    if isinstance(value, dict):
        for item in value.values():
            yield from _input_files(item, excluded)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _input_files(item, excluded)
    elif isinstance(value, (pd.DataFrame, pd.Series)):
        yield from _input_files(value.to_dict(), excluded)
    elif hasattr(value, 'filenames'):
        yield from _input_files(value.filenames, excluded)
    elif isinstance(value, (str, os.PathLike)):
        path = Path(value)
        try:
            exists = path.exists()
        except (OSError, ValueError):
            return
        if not exists:
            return
        paths = path.rglob('*') if path.is_dir() else [path]
        for file in paths:
            absolute = os.path.abspath(file)
            if (absolute == excluded or absolute.startswith(excluded + os.sep)
                    or absolute == excluded + '.lock'
                    or any(part in ('.git', '__pycache__', '.torcheeg') for part in (
                        file.relative_to(path).parts if path.is_dir() else ()))
                    or any(part.startswith('.torcheeg-cache-') for part in file.parts)):
                continue
            if file.is_file():
                yield absolute


def get_cache_key(dataset_class, configuration, records, io_path):
    """Return a key for source files, processing configuration and record order."""
    excluded = os.path.abspath(io_path)
    sources = sorted(set(_input_files((configuration, records), excluded)))
    inputs = [(path, get_file_checksum(path)) for path in sources]
    implementations = [
        getattr(dataset_class, name)
        for name in ('set_records', 'read_record', 'process_record')
    ]
    value = (dataset_class.__module__, dataset_class.__qualname__,
             _describe(configuration), _describe(records), inputs,
             _describe(implementations))
    return _get_cache_key(joblib.hash(value).encode())
