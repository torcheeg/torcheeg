Installation
====================================

TorchEEG depends on PyTorch, please complete the installation of PyTorch (>=1.8.1)
according to the system, CUDA version and other information:

.. code:: shell

   # Conda
   # please refer to https://pytorch.org/get-started/locally/
   # e.g. CPU version
   conda install pytorch==1.11.0 torchvision torchaudio cpuonly -c pytorch
   # e.g. GPU version
   conda install pytorch==1.11.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch

   # Pip
   # please refer to https://pytorch.org/get-started/previous-versions/
   # e.g. CPU version
   pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
   # e.g. GPU version
   pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu102

Anaconda
~~~~~~~~

Since version v1.0.9, torcheeg supports installing with conda! You can
simply install TorchEEG using Anaconda, just run the following command:

.. code:: shell

   conda install -c tczhangzhi -c conda-forge torcheeg

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

Plugin
~~~~~~

TorchEEG provides plugins related to graph algorithms for converting EEG
in datasets into graph structures and analyzing them using graph neural
networks. This part of the implementation relies on PyG.

If you do not use graph-related algorithms, you can skip this part of
the installation.

.. code:: shell

   # Conda
   # please refer to https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
   conda install pyg -c pyg

   # Pip
   # please refer to https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
   # e.g. CPU version
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
   # e.g. GPU version
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
