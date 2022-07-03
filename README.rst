TorchEEG
========

|PyPI Version| |Docs Status|

`Documentation <https://torcheeg.readthedocs.io/>`__ \| `TorchEEG
Examples <https://github.com/tczhangzhi/torcheeg/tree/main/examples>`__

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

Installation
------------

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

More About TorchEEG
-------------------

At a granular level, PyTorch is a library that consists of the following
components:

+----------------------------------------+-----------------------------+
| Component                              | Description                 |
+========================================+=============================+
| `torcheeg.io <https://torcheeg.readthe | A set of unified input and  |
| docs.io/en/latest/torcheeg.io.html>`__ | output API is used to store |
|                                        | the processing results of   |
|                                        | various EEG databases for   |
|                                        | more efficient and          |
|                                        | convenient use.             |
+----------------------------------------+-----------------------------+
| `torcheeg.da                           | The packaged benchmark      |
| tasets <https://torcheeg.readthedocs.i | dataset implementation      |
| o/en/latest/torcheeg.datasets.html>`__ | provides a multi-process    |
|                                        | preprocessing interface.    |
+----------------------------------------+-----------------------------+
| `torcheeg.transf                       | Extensive EEG preprocessing |
| orms <https://torcheeg.readthedocs.io/ | methods help users extract  |
| en/latest/torcheeg.transforms.html>`__ | features, construct EEG     |
|                                        | signal representations, and |
|                                        | connect to commonly used    |
|                                        | deep learning libraries.    |
+----------------------------------------+-----------------------------+
| `torcheeg.model_selection              | Extensive dataset           |
| <https://torcheeg.readthedocs.io/en/la | partitioning methods for    |
| test/torcheeg.model_selection.html>`__ | users to experiment with    |
|                                        | different settings.         |
+----------------------------------------+-----------------------------+
| `torchee                               | Extensive baseline method   |
| g.models <https://torcheeg.readthedocs | reproduction.               |
| .io/en/latest/torcheeg.models.html>`__ |                             |
+----------------------------------------+-----------------------------+

Implemented Modules
-------------------

We list currently supported datasets, transforms, data splitting, and
deep learning models by category.

**Datasets:** All datasets rely on a set of efficient IO APIs,
`torcheeg.io <https://torcheeg.readthedocs.io/en/latest/torcheeg.io.html>`__,
to store data preprocessing results on disk and read them quickly during
training. Data preprocessing and storage support multiprocessing (speed
up!).

-  `AMIGOS
   dataset <https://torcheeg.readthedocs.io/en/latest/torcheeg.datasets.html#amigosdataset>`__
   from Miranda-Correa et al.: `AMIGOS: A dataset for affect,
   personality and mood research on individuals and
   groups <https://ieeexplore.ieee.org/abstract/document/8554112/>`__.
-  `DREAMER
   dataset <https://torcheeg.readthedocs.io/en/latest/torcheeg.datasets.html#dreamerdataset>`__
   from Katsigiannis et al.: `DREAMER: A database for emotion
   recognition through EEG and ECG signals from wireless low-cost
   off-the-shelf
   devices <https://ieeexplore.ieee.org/abstract/document/7887697>`__.
-  `SEED
   dataset <https://torcheeg.readthedocs.io/en/latest/torcheeg.datasets.html#seeddataset>`__
   from Zheng et al.: `Investigating critical frequency bands and
   channels for EEG-based emotion recognition with deep neural
   networks <https://ieeexplore.ieee.org/abstract/document/7104132>`__.
-  `DEAP
   dataset <https://torcheeg.readthedocs.io/en/latest/torcheeg.datasets.html#deapdataset>`__
   from Koelstra et al.: `DEAP: A database for emotion analysis; using
   physiological
   signals <https://ieeexplore.ieee.org/abstract/document/5871728>`__.
-  `MAHNOB
   dataset <https://torcheeg.readthedocs.io/en/latest/torcheeg.datasets.html#mahnobdataset>`__
   from Soleymani et al.: `A multimodal database for affect recognition
   and implicit
   tagging <https://ieeexplore.ieee.org/abstract/document/5975141>`__.

**Transforms:** TorchEEG provides extensive data transformation tools to
help users build EEG data representations suitable for a variety of task
formulation and a variety of model structures.

-  Feature Engineering:
   `BandDifferentialEntropy <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-banddifferentialentropy>`__,
   `BandPowerSpectralDensity <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-bandpowerspectraldensity>`__,
   `BandMeanAbsoluteDeviation <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-bandmeanabsolutedeviation>`__,
   `BandKurtosis <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-bandkurtosis>`__,
   `BandSkewness <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-bandskewness>`__,
   `Concatenate <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-concatenate>`__
-  General Operation:
   `PickElectrode <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-pickelectrode>`__,
   `MeanStdNormalize <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-meanstdnormalize>`__,
   `MinMaxNormalize <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-minmaxnormalize>`__
-  For CNN:
   `To2d <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-to2d>`__,
   `ToGrid <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-togrid>`__,
   `ToInterpolatedGrid <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-tointerpolatedgrid>`__
-  For GNN:
   `ToG <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.pyg.html#transforms-tog>`__
-  For Augmentation:
   `Resize <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.torch.html#transforms-resize>`__,
   `RandomNoise <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.torch.html#transforms-randomnoise>`__,
   `RandomMask <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.torch.html#transforms-randommask>`__
-  For Label Construction:
   `Select <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.label.html#transforms-select>`__,
   `Binary <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.label.html#transforms-binary>`__,
   `BinariesToCategory <https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.label.html#transforms-binariestocategory>`__

**Data Splitting:** In current research in the field of EEG analysis,
there are various settings based on different considerations for data
partitioning. Please choose a reasonable data division method according
to the research focus:

-  Subject Dependent:
   `KFoldPerSubjectGroupbyTrial <https://torcheeg.readthedocs.io/en/latest/torcheeg.model_selection.html#kfoldpersubjectgroupbytrial>`__,
   `train_test_split_per_subject_groupby_trial <https://torcheeg.readthedocs.io/en/latest/torcheeg.model_selection.html#train-test-split-per-subject-groupby-trial>`__
-  Subject Independent:
   `LeaveOneSubjectOut <https://torcheeg.readthedocs.io/en/latest/torcheeg.model_selection.html#leaveonesubjectout>`__
-  Conventional:
   `KFold <https://torcheeg.readthedocs.io/en/latest/torcheeg.model_selection.html#kfold>`__,
   `train_test_split <https://torcheeg.readthedocs.io/en/latest/torcheeg.model_selection.html#train-test-split>`__,
   `KFoldGroupbyTrial <https://torcheeg.readthedocs.io/en/latest/torcheeg.model_selection.html#kfoldgroupbytrial>`__,
   `train_test_split_groupby_trial <https://torcheeg.readthedocs.io/en/latest/torcheeg.model_selection.html#train-test-split-groupby-trial>`__

**Models:** Coming soon after pushing to align with the official
implementation or description. If the current version of
`CNNs <https://torcheeg.readthedocs.io/en/latest/torcheeg.models.cnn.html>`__,
`GNNs <https://torcheeg.readthedocs.io/en/latest/torcheeg.models.gnn.html>`__
and
`Transformers <https://torcheeg.readthedocs.io/en/latest/torcheeg.models.transformer.html>`__
is to be used, please refer to the implementation in
`torcheeg.models <https://torcheeg.readthedocs.io/en/latest/torcheeg.models.html>`__.

Quickstart
----------

In this quick tour, we highlight the ease of starting an EEG analysis
research with only modifying a few lines of `PyTorch
tutorial <https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html>`__.

The ``torcheeg.datasets`` module contains dataset classes for many
real-world EEG datasets. In this tutorial, we use the ``DEAP`` dataset.
We first go to the official website to apply for data download
permission according to the introduction of `DEAP
dataset <https://www.eecs.qmul.ac.uk/mmv/datasets/deap/>`__, and
download the dataset. Next, we need to specify the download location of
the dataset in the ``root_path`` parameter. For the DEAP dataset, we
specify the path to the ``data_preprocessed_python`` folder,
e.g. ``./tmp_in/data_preprocessed_python``.

.. code:: python

   from torcheeg.datasets import DEAPDataset
   from torcheeg.datasets.constants.emotion_recognition.deap import DEAP_CHANNEL_LOCATION_DICT

   dataset = DEAPDataset(io_path=f'./tmp_out/deap',
                         root_path='./tmp_in/data_preprocessed_python',
                         offline_transform=transforms.Compose(
                             [transforms.BandDifferentialEntropy(),
                              transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)]),
                         online_transform=transforms.Compose([transforms.BaselineRemoval(),
                                                              transforms.ToTensor()]),
                         label_transform=transforms.Compose([
                             transforms.Select('valence'),
                             transforms.Binary(5.0),
                         ]), num_worker=4)

The ``DEAPDataset`` API further contains three parameters:
``online_transform``, ``offline_transform``, and ``label_transform``,
which are used to modify samples and labels, respectively.

Here, ``offline_transform`` will only be called once when the dataset is
initialized to preprocess all samples in the dataset, and the processed
dataset will be stored in ``io_path`` to avoid time-consuming repeated
transformations in subsequent use. If offline preprocessing is a
computationally intensive operation, we also recommend setting multi-CPU
parallelism for offline_transform, e.g., set ``num_worker`` to 4.

``online_transform`` is used to transform samples on the fly. Please use
``online_transform`` if you don’t want to wait for the preprocessing of
the entire dataset (suitable for scenarios where new ``transform``
algorithms are designed) or expect data transformation with randomness
each time a sample is indexed.

Next, we need to divide the dataset into a training set and a test set.
In the field of EEG analysis, commonly used data partitioning methods
include k-fold cross-validation and leave-one-out cross-validation. In
this tutorial, we use k-fold cross-validation on the entire dataset
(``KFold``) as an example of dataset splitting.

.. code:: python

   from torcheeg.model_selection import KFold

   k_fold = KFold(n_splits=10,
                         split_path=f'./tmp_out/split',
                         shuffle=True,
                         random_state=42)

Let's define a simple but effective CNN model according to
`CCNN <https://link.springer.com/chapter/10.1007/978-3-030-04239-4_39>`__:

.. code:: python

   class CNN(torch.nn.Module):
       def __init__(self, in_channels=4, num_classes=3):
           super().__init__()
           self.conv1 = nn.Sequential(
               nn.ZeroPad2d((1, 2, 1, 2)),
               nn.Conv2d(in_channels, 64, kernel_size=4, stride=1),
               nn.ReLU()
           )
           self.conv2 = nn.Sequential(
               nn.ZeroPad2d((1, 2, 1, 2)),
               nn.Conv2d(64, 128, kernel_size=4, stride=1),
               nn.ReLU()
           )
           self.conv3 = nn.Sequential(
               nn.ZeroPad2d((1, 2, 1, 2)),
               nn.Conv2d(128, 256, kernel_size=4, stride=1),
               nn.ReLU()
           )
           self.conv4 = nn.Sequential(
               nn.ZeroPad2d((1, 2, 1, 2)),
               nn.Conv2d(256, 64, kernel_size=4, stride=1),
               nn.ReLU()
           )

           self.lin1 = nn.Linear(9 * 9 * 64, 1024)
           self.lin2 = nn.Linear(1024, num_classes)

       def forward(self, x):
           x = self.conv1(x)
           x = self.conv2(x)
           x = self.conv3(x)
           x = self.conv4(x)

           x = x.flatten(start_dim=1)
           x = self.lin1(x)
           x = self.lin2(x)
           return x

Specify the device and loss function used during training and test.

.. code:: python

   device = "cuda" if torch.cuda.is_available() else "cpu"
   loss_fn = nn.CrossEntropyLoss()
   batch_size = 64

The training and validation scripts for the model are taken from the
`PyTorch
tutorial <https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html>`__
without much modification. Usually, the value of ``batch`` contains two
parts; the first part refers to the result of ``online_transform``,
which generally corresponds to the ``Tensor`` sequence representing EEG
signals. The second part refers to the result of ``label_transform``, a
sequence of integers representing the label.

.. code:: python

   def train(dataloader, model, loss_fn, optimizer):
       size = len(dataloader.dataset)
       model.train()
       for batch_idx, batch in enumerate(dataloader):
           X = batch[0].to(device)
           y = batch[1].to(device)

           # Compute prediction error
           pred = model(X)
           loss = loss_fn(pred, y)

           # Backpropagation
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           if batch_idx % 100 == 0:
               loss, current = loss.item(), batch_idx * len(X)
               print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


   def valid(dataloader, model, loss_fn):
       size = len(dataloader.dataset)
       num_batches = len(dataloader)
       model.eval()
       val_loss, correct = 0, 0
       with torch.no_grad():
           for batch in dataloader:
               X = batch[0].to(device)
               y = batch[1].to(device)

               pred = model(X)
               val_loss += loss_fn(pred, y).item()
               correct += (pred.argmax(1) == y).type(torch.float).sum().item()
       val_loss /= num_batches
       correct /= size
       print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")

Traverse ``k`` folds and train the model separately for testing. It is
worth noting that, in general, we need to specify ``shuffle=True`` for
the ``DataLoader`` of the training data set to avoid the deviation of
the model training caused by consecutive labels of the same category.

.. code:: python

   for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):

       model = CNN().to(device)
       optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

       train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
       val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

       epochs = 50
       for t in range(epochs):
           print(f"Epoch {t+1}\n-------------------------------")
           train(train_loader, model, loss_fn, optimizer)
           valid(val_loader, model, loss_fn)
       print("Done!")

For more specific usage of each module, please refer to `the
documentation <(https://torcheeg.readthedocs.io/)>`__.

Releases and Contributing
-------------------------

TorchEEG is currently in beta; Please let us know if you encounter a bug
by filing an issue. We also appreciate all contributions.

If you would like to contribute new datasets, deep learning methods, and
extensions to the core, please first open an issue and then send a PR.
If you are planning to contribute back bug fixes, please do so without
any further discussion.

About Us
--------

The following authors provide long-term support for this project. If you
notice anything in the project that is not as expected, please do not
hesitate to contact us.

`Zhi ZHANG <mailto:tczhangzhi@gmail.com>`__: received the M.Eng. degree
at the College of Computer Science and Software Engineering from
Shenzhen University, China, in 2021. He is currently with the Hong Kong
Polytechnic University as a PhD candidate. His research interests mainly
include graph convolutional networks, abnormal event detection, and EEG
analysis.

`Sheng-hua ZHONG <mailto:csshzhong@szu.edu.cn>`__: received the
Ph.D. degree from the Department of Computing, The Hong Kong Polytechnic
University in 2013. Currently, she is an Associate Professor in College
of Computer Science & Software Engineering at Shenzhen University. Her
research interests include multimedia content analysis and brain
science.

`Yan LIU <mailto:csyliu@comp.polyu.edu.hk>`__: is the director of
cognitive computing lab and the group leader of artificial intelligence
and robotics AIR research group. She obtained Ph.D. degree in computer
Science from Columbia University in the US. In 2005, she joined The Hong
Kong Polytechnic University, Hong Kong, where she is currently an
Associate Professor with the Department of Computing. Her research
interests span a wide range of topics, ranging from brain modeling and
cognitive computing, image/video retrieval, computer music to machine
learning and pattern recognition.

License
-------

TorchEEG has a MIT license, as found in the
`LICENSE <https://github.com/tczhangzhi/torcheeg/blob/main/LICENSE>`__
file.

.. |PyPI Version| image:: https://badge.fury.io/py/torcheeg.svg
   :target: https://pypi.python.org/pypi/torcheeg
.. |Docs Status| image:: https://readthedocs.org/projects/torcheeg/badge/?version=latest
   :target: https://torcheeg.readthedocs.io/en/latest/?badge=latest
