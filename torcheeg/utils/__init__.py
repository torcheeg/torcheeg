from importlib import import_module as _import_module


_EXPORTS = {
    'seed_everything': '.experiment',
    'download_url': '.file',
    'get_package_dir_path': '.file',
    'get_random_dir_path': '.file',
    'get_temp_dir_path': '.file',
    'plot_2d_tensor': '.visualize',
    'plot_3d_tensor': '.visualize',
    'plot_adj_connectivity': '.visualize',
    'plot_feature_topomap': '.visualize',
    'plot_graph': '.visualize',
    'plot_raw_topomap': '.visualize',
    'plot_signal': '.visualize',
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
