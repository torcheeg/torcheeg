Installation
====================================

TorchEEG depends on PyTorch, please complete the installation of PyTorch
according to the system, CUDA version and other information:

.. code:: shell

   # please refer to https://pytorch.org/get-started/previous-versions/
   # e.g. CPU version
   pip install torch==1.10.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
   # e.g. GPU version
   pip install torch==1.10.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html

TorchEEG provides algorithms related to graph convolution. This part of
the implementation relies on PyG. TorchEEG recommends users to manually
install PyG to avoid possible errors:

.. code:: shell

   # please refer to https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
   # e.g. CPU version
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
   # e.g. GPU version
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu102.html

Pip
~~~

TorchEEG allows pip-based installation, please use the following
command:

.. code:: shell

   pip install torcheeg

Nightly
~~~~~~~

In case you want to experiment with the latest TorchEEG features which
are not fully released yet, please run the following command to install
from the main branch on github:

.. code:: shell

   pip install git+https://github.com/tczhangzhi/torcheeg.git

