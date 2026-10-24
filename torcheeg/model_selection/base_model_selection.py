import os
from typing import Dict, Union

import pandas as pd

from ..utils import get_random_dir_path
from .utils import ensure_cache, validate_metadata


class BaseModelSelection:
    r'''Common initialization, checks and cache access for dataset splitters.'''

    def __init__(self, split_path: Union[None, str] = None) -> None:
        self.split_path = (get_random_dir_path(dir_prefix='model_selection')
                           if split_path is None else os.fspath(split_path))

    def build(self, info: pd.DataFrame, split_path: Union[None, str] = None) -> None:
        r'''Write the row positions for this splitting rule.'''
        raise NotImplementedError("Method build is not implemented in class " +
                                  self.__class__.__name__)

    def split(self, dataset, **kwargs):
        r'''Yield dataset subsets for this splitting rule.'''
        raise NotImplementedError("Method split is not implemented in class " +
                                  self.__class__.__name__)

    def check(self, dataset, columns=()):
        r'''Check the metadata required by a splitting rule.'''
        validate_metadata(dataset.info, columns)

    def _prepare(self, dataset, columns=(), files=None, configuration=None):
        self.check(dataset, columns)
        parameters = dict(self.repr_body)
        parameters.pop('split_path', None)
        parameters['splitter'] = self.__class__.__name__
        if configuration is not None:
            parameters.update(configuration)
        ensure_cache(dataset.info, columns, self.split_path, parameters, files,
                     lambda path: self.build(dataset.info, path))

    @property
    def repr_body(self) -> Dict:
        return {'split_path': self.split_path}

    def __repr__(self) -> str:
        values = ', '.join(f'{name}={value!r}'
                           for name, value in self.repr_body.items())
        return f'{self.__class__.__name__}({values})'
