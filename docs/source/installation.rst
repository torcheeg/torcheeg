Install TorchEEG
================

Install TorchEEG in an isolated Python environment. Select a Python version
supported by the PyTorch build that you need.

Install PyTorch
---------------

Use the `official PyTorch installation selector
<https://pytorch.org/get-started/locally/>`_ to select your operating system,
package manager, and CPU or accelerator build. For a specific older build,
use the `previous-version instructions
<https://pytorch.org/get-started/previous-versions/>`_.

Install a PyTorch build that matches your Python version and hardware before
installing TorchEEG. Graph extensions also need to match that build.

Install a release
-----------------

Use pip to install the published package:

.. code-block:: shell

   python -m pip install torcheeg

The installed release and the development branch can expose different APIs.
Use the documentation for the version that you installed.

Conda packages
--------------

TorchEEG also has a package in the ``tczhangzhi`` conda channel:

.. code-block:: shell

   conda install -c tczhangzhi -c conda-forge torcheeg

Check the package version in the channel before selecting an API. A conda
package and the PyPI package can be published at different times.

Install from source
-------------------

Use a source checkout to work with changes that are not in a published release:

.. code-block:: shell

   git clone https://github.com/torcheeg/torcheeg.git
   cd torcheeg
   python -m pip install -e .

Use a separate checkout when you need to keep an existing working directory.
A source installation uses the code in that checkout.

Optional components
-------------------

Install the dependency group needed for your workflow:

.. list-table::
   :header-rows: 1
   :widths: 20 50

   * - Group
     - Purpose
   * - ``transforms``
     - Optional signal resampling dependencies.
   * - ``dataset``
     - MOABB integration.
   * - ``trainer``
     - Optional metric dependencies used by generative trainers.
   * - ``model``
     - PyTorch Geometric and graph extensions.
   * - ``test``
     - Development test dependencies.

For example:

.. code-block:: shell

   python -m pip install "torcheeg[transforms]"

For a source checkout, use ``python -m pip install -e '.[transforms]'``.
You can combine groups, for example ``torcheeg[dataset,transforms]``.

Before installing the graph group, follow the `PyTorch Geometric installation
guide <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_
to select extension wheels for your PyTorch version and CPU or accelerator
build. Reusing extension wheels from another PyTorch build can fail at import.

Check the installation
----------------------

Check the installed versions and dependency requirements:

.. code-block:: shell

   python -c "import torch, torcheeg; print(torch.__version__, torcheeg.__version__)"
   python -m pip check

Continue with the :doc:`user_guide/index` and :doc:`tutorials/index`.

SciPy versions
--------------

The development version requires SciPy 1.7.3 or later. The previous upper
limit of 1.10.1 is removed. Select a SciPy version that supports your Python
version. This change does not add new APIs to older TorchEEG releases.

Use examples from a source archive
----------------------------------

Source archives built from this checkout include standalone Python scripts
in ``examples/``. Unpack the source archive, install the library and the
dependency groups required by the script, and run it from the source root.
Check the script for dataset download requirements and cache paths first.
The :doc:`tutorials/index` page links the documented workflows.
