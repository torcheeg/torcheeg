Install TorchEEG
================

TorchEEG is a Python 3 library for EEG analysis with PyTorch.
Use Python 3.10 for the installation steps on this page.

The package is available on `PyPI <https://pypi.org/project/torcheeg/>`_
and `Anaconda <https://anaconda.org/tczhangzhi/torcheeg>`_.
The source code is available on `GitHub
<https://github.com/torcheeg/torcheeg>`_.

Choose pip for a release, conda for a managed environment, or source code
for development.

Install PyTorch
---------------

Install PyTorch before you install TorchEEG.
Use the `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_
to select your operating system and compute platform.
Run its installation command in your Python environment.

For an NVIDIA GPU:

1. Install an NVIDIA driver that supports your selected PyTorch CUDA build.
2. Select the CUDA build in the PyTorch installation guide.
3. Run the installation command from the guide.

Install with pip
----------------

Install the release from PyPI:

.. code-block:: shell

   python -m pip install torcheeg

Add the ``moabb`` plugin to load public EEG datasets through
`MOABB <https://moabb.neurotechx.com/>`_:

.. code-block:: shell

   python -m pip install "torcheeg[moabb]"

Add the ``pyg`` plugin to use graph transforms and graph neural networks.
First, follow the `PyTorch Geometric installation guide
<https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_
to install extension wheels for your PyTorch version and compute platform.
Then install the plugin:

.. code-block:: shell

   python -m pip install "torcheeg[pyg]"

Combine plugin names to install both:

.. code-block:: shell

   python -m pip install "torcheeg[pyg,moabb]"

Install with conda
------------------

Create and activate a Python environment:

.. code-block:: shell

   conda create -n torcheeg python=3.10 pip
   conda activate torcheeg

Install PyTorch in this environment with the command from its installation
guide. Then install TorchEEG:

.. code-block:: shell

   conda install -c tczhangzhi -c conda-forge torcheeg

Use pip in the active conda environment to add plugins:

.. code-block:: shell

   python -m pip install "torcheeg[moabb]"

For graph support, install the matching PyG extension wheels as described
above. Then run:

.. code-block:: shell

   python -m pip install "torcheeg[pyg]"

Install from source
-------------------

Install the development version directly from GitHub:

.. code-block:: shell

   python -m pip install "git+https://github.com/torcheeg/torcheeg.git@main"

To edit the code, clone the repository and open its directory:

.. code-block:: shell

   git clone https://github.com/torcheeg/torcheeg.git
   cd torcheeg

Install the local source in editable mode:

.. code-block:: shell

   python -m pip install -e .

Python then uses the source files in this directory.
Your code changes take effect when you start a new Python process.

The development version uses these dependency groups:

* ``model``: PyG graph transforms and graph neural networks.
* ``dataset``: MOABB datasets.
* ``transforms``: signal resampling with soxr.
* ``trainer``: optional metrics for generative trainers.
* ``test``: development test tools.

For MOABB support:

.. code-block:: shell

   python -m pip install -e ".[dataset]"

For graph support, install the matching PyG extension wheels. Then run:

.. code-block:: shell

   python -m pip install -e ".[model]"

Install all development groups for the test suite:

.. code-block:: shell

   python -m pip install -e ".[test,model,trainer,transforms,dataset]"

Install the documentation dependencies to build the documentation:

.. code-block:: shell

   python -m pip install -r docs/requirements.txt

Check the installation
----------------------

Run this command to display the installed versions:

.. code-block:: shell

   python -c "import torch, torcheeg; print(torch.__version__, torcheeg.__version__)"

Check the dependency requirements:

.. code-block:: shell

   python -m pip check

Continue with the :doc:`user_guide/index` or :doc:`tutorials/index`.
See the :doc:`contributing` for development and testing instructions.
