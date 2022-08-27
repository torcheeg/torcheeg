try:
    import torch_geometric
except ImportError as e:
    raise ImportError(
        "To run the torcheeg with graph neural networks and related transforms, you need to install `torch_geometric`. Please refer to `https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html`."
    ) from e

from .to import *