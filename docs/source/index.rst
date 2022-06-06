.. torcheeg documentation master file, created by
   sphinx-quickstart on Mon May 30 00:14:49 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TorchEEG's documentation!
====================================

TorchEEG is a library built on PyTorch for EEG signal analysis. TorchEEG
aims to provide a plug-and-play EEG analysis tool, so that researchers
can quickly reproduce EEG analysis work and start new EEG analysis
research without paying attention to technical details unrelated to the
research focus.

TorchEEG specifies a unified data input-output format (IO) and implement
commonly used EEG databases, allowing users to quickly access benchmark
datasets and define new custom datasets. The datasets that have been
defined so far include emotion recognition and so on. According to
papers published in the field of EEG analysis, TorchEEG provides data
preprocessing methods commonly used for EEG signals, and provides
plug-and-play API for both offline and online pre-proocessing. Offline
processing allow users to process once and use any times, speeding up
the training process. Online processing allows users to save time when
creating new data processing methods. TorchEEG also provides deep
learning models following published papers for EEG analysis, including
convolutional neural networks, graph convolutional neural networks, and
Transformers.

.. toctree::
   :maxdepth: 1
   :caption: Notes:

   installation
   introduction_by_example
   creating_your_own_datasets

.. toctree::
   :maxdepth: 2
   :caption: Package Reference:

   torcheeg.datasets
   torcheeg.io
   torcheeg.model_selection
   torcheeg.models
   torcheeg.transforms

Indices and tables
==================

* :ref:`search`
