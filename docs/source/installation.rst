Installation
====================================

TorchEEG depends on PyTorch, please complete the installation of PyTorch
according to the system, CUDA version and other information:

.. code:: shell

   # please refer to https://pytorch.org/get-started/previous-versions/
   # e.g. CPU version
   pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
   # e.g. GPU version
   pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu102

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

