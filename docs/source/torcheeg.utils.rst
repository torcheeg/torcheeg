torcheeg.utils
==============

Shared utilities live in the single ``torcheeg/utils.py`` module. Utilities that
belong to a specific domain remain in that domain's existing ``utils.py``.
For plotting APIs, see :doc:`torcheeg.visualizers`.

Files and experiments
---------------------

.. autofunction:: torcheeg.utils.seed_everything

.. autofunction:: torcheeg.utils.download_url

.. autofunction:: torcheeg.utils.get_package_dir_path

.. autofunction:: torcheeg.utils.get_random_dir_path

.. autofunction:: torcheeg.utils.get_temp_dir_path

Shared cache operations
-----------------------

Dataset and split-specific fingerprinting stays in ``datasets.utils`` and
``model_selection.utils``. These functions provide their shared cache machinery.

.. autofunction:: torcheeg.utils.compute_fingerprint

.. autofunction:: torcheeg.utils.compute_checksum

.. autofunction:: torcheeg.utils.validate_cache

.. autofunction:: torcheeg.utils.ensure_cache

Optional dependencies
---------------------

.. autoclass:: torcheeg.utils.LazyLoader

``LazyLoader`` loads its module on first attribute access. Resolving a class as
a base also accesses an attribute, so a top-level class definition triggers the
import at that point. Importing shared utilities does not load visualizers.

Migration
---------

Replace direct
imports from the former ``torcheeg.utils.file``, ``torcheeg.utils.experiment``,
``torcheeg.utils.cache`` and ``torcheeg.utils.lazy_loader`` submodules with imports
from ``torcheeg.utils``. For example::

    from torcheeg.utils import LazyLoader, download_url, ensure_cache

Plotting functions now live in ``torcheeg.visualizers``. Replace plotting imports
from ``torcheeg.utils`` or ``torcheeg.utils.visualize`` with imports from
``torcheeg.visualizers``. No domain-specific ``utils.py`` file is renamed or moved.
