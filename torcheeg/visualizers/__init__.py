from importlib import import_module as _import_module


_EXPORTS = {
    'plot_signal': '.raw',
    'plot_raw_topomap': '.topomap',
    'plot_feature_topomap': '.topomap',
    'plot_3d_tensor': '.tensor',
    'plot_2d_tensor': '.tensor',
    'plot_adj_connectivity': '.connectivity',
    'plot_graph': '.graph',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
