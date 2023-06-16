try:
    import moabb
except ImportError:
    raise ImportError(
        "MOABB is not installed. Please install it by 'pip install moabb'.")

from .moabb_dataset import MOABBDataset