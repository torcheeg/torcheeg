Creating Your Own Datasets
====================================

Here, we describe how to define a TorchEEG Dataset to apply
the convenient utility functions in TorchEEG on arbitrary data formats.

TorchEEG provides a unified data IO format for all EEG databases. When
using data from different sources, we only need to convert it to the
unified data IO format to be compatible with subsequent operations and
modules provided by TorchEEG. Don't worry. Converting to a unified data
storage format is quite straightforward. TorchEEG has already handled
the content related to data IO reading and writing for you, you only
need to care about how to define each EEG signal sample and its
corresponding label information.

Lets's start!

First, import ``BaseDataset`` from ``torcheeg.datasets`` and define a
custom database inheriting ``BaseDataset``. The methods extending
``BaseDataset`` will automatically handle the already generated IO .

::

   class CustomDataset(BaseDataset):
       ...

Next, we define how to generate data IO in the constructor:

::

   from torcheeg.datasets import BaseDataset

   class CustomDataset(BaseDataset):
       def __init__(self, io_path):
           self.dataset_constructor(io_path)
           super().__init__(io_path)

       def dataset_constructor(self, io_path):
           ...

Here, two tool classes ``EEGSignalIO`` and ``MetaInfoIO`` can be very
helpful. ``EEGSignalIO`` is designed to build a small and
efficientdatabase to read and write EEG signals (like
`caffe2 <https://caffe2.ai/>`__). ``MetaInfoIO`` is used to build a
meta-information file to read and write and descriptive information for
signal samples. It is recommended to use ``eeg`` and ``info.csv`` as the
names to store the them in the ``io_path`` folder:

::

   from torcheeg.datasets import BaseDataset
   from torcheeg.io import EEGSignalIO, MetaInfoIO

   class CustomDataset(BaseDataset):
       def __init__(self, io_path):
           self.dataset_constructor(io_path)
           super().__init__(io_path)

       def dataset_constructor(self, io_path):
           os.makedirs(io_path)

           meta_info_io_path = os.path.join(io_path, 'info.csv')
           eeg_signal_io_path = os.path.join(io_path, 'eeg')

           info_io = MetaInfoIO(meta_info_io_path)
           eeg_io = EEGSignalIO(eeg_signal_io_path)

Next, get a sample of the EEG signal depending on the situation (maybe
you need to iterate over your recorded EEG file) and write it to the EEG
database using ``eeg_io.write_eeg(...)``. Since many algorithms also
rely on the baseline signal, the baseline signal usually needs to be
written into the EEG database too. TorchEEG recommends using
``np.ndarray`` format as input to store EEG signals.

For labels corresponding to EEG samples, we use
``info_io.write_info(...)`` to store them in the meta information file.
TorchEEG recommends storing as much description information of EEG
samples as possible for future use, and using transforms to filter them
if necessary.

::

   from torcheeg.datasets import BaseDataset
   from torcheeg.io import EEGSignalIO, MetaInfoIO

   class CustomDataset(BaseDataset):
       def __init__(self, io_path):
           self.dataset_constructor(io_path)
           super().__init__(io_path)

       def dataset_constructor(self, io_path):
           os.makedirs(io_path)

           meta_info_io_path = os.path.join(io_path, 'info.csv')
           eeg_signal_io_path = os.path.join(io_path, 'eeg')

           info_io = MetaInfoIO(meta_info_io_path)
           eeg_io = EEGSignalIO(eeg_signal_io_path)

           trail_baseline_sample = ... # np.random.randn(32, 128)
           transformed_eeg = self.offline_transform(trail_baseline_sample)
           trail_base_id = eeg_io.write_eeg(transformed_eeg)

           clip_sample = ... # np.random.randn(32, 128)
           transformed_eeg = self.offline_transform(clip_sample)
           clip_id = eeg_io.write_eeg(transformed_eeg)

           label_info = ... # {'valence': 1.0, 'arousal': 3.0}
           clip_info = {'baseline_id': trail_base_id, 'clip_id': clip_id, **label_info}
           info_io.write_info(clip_info)

Finally, define the ``__getitem__`` function to declare the return
information when traversing the ``Dataset``. TorchEEG provides
``self.info`` and ``self.eeg_io`` primitives for obtaining meta
information and EEG signals:

::

   from torcheeg.datasets import BaseDataset
   from torcheeg.io import EEGSignalIO, MetaInfoIO

   class CustomDataset(BaseDataset):
       def __init__(self, io_path, online_transform=None, offline_transform=None, label_transform=None): # other parameters
           self.dataset_constructor() # other parameters
           super().__init__(io_path)
           self.online_transform = online_transform
           self.offline_transform = offline_transform
           self.label_transform = label_transform

       def dataset_constructor(self):
           os.makedirs(self.io_path)

           meta_info_io_path = os.path.join(self.io_path, 'info.csv')
           eeg_signal_io_path = os.path.join(self.io_path, 'eeg')

           info_io = MetaInfoIO(meta_info_io_path)
           eeg_io = EEGSignalIO(eeg_signal_io_path)

           trail_baseline_sample = ... # np.random.randn(32, 128)
           transformed_eeg = self.offline_transform(trail_baseline_sample)
           trail_base_id = eeg_io.write_eeg(transformed_eeg)

           clip_sample = ... # np.random.randn(32, 128)
           transformed_eeg = self.offline_transform(clip_sample)
           clip_id = eeg_io.write_eeg(transformed_eeg)

           label_info = ... # {'valence': 1.0, 'arousal': 3.0}
           clip_info = {'baseline_id': trail_base_id, 'clip_id': clip_id, **label_info}
           info_io.write_info(clip_info)
           
       def __getitem__(self, index):
           info = self.info.iloc[index].to_dict()

           eeg_index = str(info['clip_id'])
           eeg = self.eeg_io.read_eeg(eeg_index)

           if self.online_transform:
               eeg = self.online_transform(eeg)

           baseline_index = str(info['baseline_id'])
           baseline = self.eeg_io.read_eeg(baseline_index)

           if self.online_transform:
               baseline = self.online_transform(baseline)

           if self.label_transform:
               info = self.label_transform(info)

           return eeg, baseline, info

For meta information, the best practice is to use
``self.info.iloc[index].to_dict()`` to convert the meta information of
the sample to a ``dict``, which is then processed by
``label_transform``. For EEG samples, the best practice is to use
``self.eeg_io.read_eeg(...)`` to obtain the ``clip_id`` in the meta
information, and pass it to ``online_transform`` for subsequent
processing, and the same is true for the baseline signal.

So far, we have completed a new database definition. Don't hesitate to
use it to open new experiments and validations! In the meantime, feel
free to submit PRs and contribute your database to TorchEEG.

