torcheeg.visualizers
================================

Plot EEG signals, scalp maps, tensors and connectivity through a dedicated
visualization API. Import a function from the package or its functional module::

    from torcheeg.visualizers import plot_signal, plot_feature_topomap
    from torcheeg.visualizers.tensor import plot_2d_tensor

Functions keep their existing arguments and return image arrays. Plotting modules
are imported on demand: importing a tensor plot does not import graph plotting.
Graph plotting requires the optional graph dependencies when used.

.. contents:: Visualization functions
    :depth: 2
    :local:
    :backlinks: top

Raw signals
-----------

Module: ``torcheeg.visualizers.raw``.

.. autofunction:: torcheeg.visualizers.plot_signal

Scalp topographies
------------------

Module: ``torcheeg.visualizers.topomap``. Raw-signal topographies show selected
time points; feature topographies show the supplied per-electrode features.

.. autofunction:: torcheeg.visualizers.plot_raw_topomap

.. autofunction:: torcheeg.visualizers.plot_feature_topomap

Tensors
-------

Module: ``torcheeg.visualizers.tensor``.

.. autofunction:: torcheeg.visualizers.plot_2d_tensor

.. autofunction:: torcheeg.visualizers.plot_3d_tensor

Connectivity
------------

Module: ``torcheeg.visualizers.connectivity``.

.. autofunction:: torcheeg.visualizers.plot_adj_connectivity

Graph networks
--------------

Module: ``torcheeg.visualizers.graph``.

.. autofunction:: torcheeg.visualizers.plot_graph

Migration
---------

Replace plotting imports from ``torcheeg.utils`` or ``torcheeg.utils.visualize``
with imports from ``torcheeg.visualizers`` or the functional modules above.

This reorganization moves the existing seven functions. It does not add a
spectrum/PSD plotting function. Frequency-band feature maps can still be drawn
with ``plot_feature_topomap`` after computing the features separately.
