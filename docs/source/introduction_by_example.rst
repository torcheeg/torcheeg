Introduction by Example
=======================

In this quick tour, we highlight the ease of starting an EEG analysis
research with only modifying a few lines of `PyTorch
tutorial <https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html>`__.

Define the Dataset
~~~~~~~~~~~~~~~~~~

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
                             [transforms.BandDifferentialEntropy(apply_to_baseline=True),
                              transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True)]),
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
``online_transform`` if you don't want to wait for the preprocessing of
the entire dataset (suitable for scenarios where new ``transform``
algorithms are designed) or expect data transformation with randomness
each time a sample is indexed.

For more datasets, please refer to
https://torcheeg.readthedocs.io/en/latest/torcheeg.datasets.html.

Define the Data Splitting Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we need to divide the dataset into a training set and a test set.
In the field of EEG analysis, commonly used data partitioning methods
include k-fold cross-validation and leave-one-out cross-validation. In
this tutorial, we use k-fold cross-validation on the entire dataset
(``KFoldDataset``) as an example of dataset splitting.

.. code:: python

   from torcheeg.model_selection import KFoldDataset

   k_fold = KFoldDataset(n_splits=10,
                         split_path=f'./tmp_out/split',
                         shuffle=True,
                         random_state=42)

For more data splitting methods, please refer to
https://torcheeg.readthedocs.io/en/latest/torcheeg.model_selection.html

Define the Model
~~~~~~~~~~~~~~~~

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

For more models, please refer to
https://torcheeg.readthedocs.io/en/latest/torcheeg.models.html

Define the Training and Test Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify the device and loss function used during training and test.

.. code:: python

   device = "cuda" if torch.cuda.is_available() else "cpu"
   loss_fn = nn.CrossEntropyLoss()
   batch_size = 64

The training and validation scripts for the model are taken from the
`PyTorch
tutorial <https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html>`__
without much modification. Usually, the value of ``batch``
contains two parts; the first part refers to the result of
``online_transform``, which generally corresponds to the ``Tensor``
sequence representing EEG signals. The second part refers to the result
of ``label_transform``, a sequence of integers representing the label.

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

For full code, please refer to
https://github.com/tczhangzhi/torcheeg/blob/main/examples/examples_torch.py.