from importlib import import_module


class LazyLoader:
    def __init__(self, lib_name):
        self._lib_name = lib_name
        self._module = None

    def __getattr__(self, name):
        if self._module is None:
            try:
                self._module = import_module(self._lib_name)
            except ImportError as error:
                raise ImportError(
                    f"To use this functionality, you need to install `{self._lib_name}`. "
                    f"Please refer to https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html"
                ) from error
        return getattr(self._module, name)
