Implementing GNNs with PyG
==========================

In this quick tour, we'll take a closer look at how to bring together
TorchEEG and `PyG
(pytorch_geometric) <https://github.com/pyg-team/pytorch_geometric>`__
to implement graph convolutional networks.

Define the Dataset
~~~~~~~~~~~~~~~~~~

The ``torcheeg.datasets`` module contains dataset classes for many
real-world EEG datasets. In this tutorial, we use the ``SEED`` dataset.
We first go to the official website to apply for data download
permission according to the introduction of `SEED
dataset <https://bcmi.sjtu.edu.cn/home/seed/>`__, and download the
dataset. Next, we need to specify the download location of the dataset
in the ``root_path`` parameter. For the SEED dataset, we specify the
path to the ``Preprocessed_EEG`` folder,
e.g. ``./tmp_in/Preprocessed_EEG``.

.. code:: python

   from torcheeg.datasets import SEEDDataset
   from torcheeg.datasets.constants.emotion_recognition.seed import \
       SEED_ADJACENCY_MATRIX

   dataset = SEEDDataset(io_path=f'./tmp_out/seed',
                         root_path='./tmp_in/Preprocessed_EEG',
                         offline_transform=transforms.BandDifferentialEntropy(apply_to_baseline=True),
                         online_transform=transforms.Compose([
                             transforms.BaselineRemoval(),
                             transforms.ToG(SEED_ADJACENCY_MATRIX)
                         ]),
                         label_transform=transforms.Compose([
                             transforms.Select('emotion'),
                             transforms.Lambda(lambda x: x + 1),
                         ]),
                         num_worker=4)

The ``SEEDDataset`` API further contains three parameters:
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

To convert raw data in numpy format into a graph representation
acceptable to PyG (``torch_geometric.data.Data``), TorchEEG provides the
``transforms.ToG`` . Here, electrodes correspond to nodes in the graph
structure, and the associations between electrodes are defined as edges
and weights on edges. The commonly considered associations are spatial
adjacency and functional connection. Here, we use the adjacency matrix
``SEED_ADJACENCY_MATRIX`` defined based on the spatial neighbor
relationship of electrodes. Each value in the adjacency matrix indicates
whether two corresponding electrodes are adjacent in a 10-20 system, 1
for adjacent and 0 for non-adjacent.

Define the Data Splitting Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we need to divide the dataset into a training set and a test set.
In the field of EEG analysis, commonly used data partitioning methods
include k-fold cross-validation and leave-one-out cross-validation. In
this tutorial, we use k-fold cross-validation per subject
(``KFoldTrialPerSubject``) as an example of dataset splitting.

.. code:: python

   from torcheeg.model_selection import KFoldDataset

   k_fold = KFoldTrialPerSubject(n_splits=10,
                                 split_path=f'./tmp_out/split',
                                 shuffle=False)

For more data splitting methods, please refer to
https://torcheeg.readthedocs.io/en/latest/torcheeg.model_selection.html

Define the Model
~~~~~~~~~~~~~~~~

Let's define a simple but effective GNN model based on the convolutional
layers and operation provided by PyG:

.. code:: python

   from torch_geometric.nn import GATConv, global_mean_pool

   class GNN(torch.nn.Module):
       def __init__(self, in_channels=4, num_layers=3, hid_channels=64, num_classes=3):
           super().__init__()
           self.conv1 = GATConv(in_channels, hid_channels)
           self.convs = torch.nn.ModuleList()
           for _ in range(num_layers - 1):
               self.convs.append(GATConv(hid_channels, hid_channels))
           self.lin1 = Linear(hid_channels, hid_channels)
           self.lin2 = Linear(hid_channels, num_classes)

       def reset_parameters(self):
           self.conv1.reset_parameters()
           for conv in self.convs:
               conv.reset_parameters()
           self.lin1.reset_parameters()
           self.lin2.reset_parameters()

       def forward(self, data):
           x, edge_index, batch = data.x, data.edge_index, data.batch
           x = F.relu(self.conv1(x, edge_index))
           for conv in self.convs:
               x = F.relu(conv(x, edge_index))
           x = global_mean_pool(x, batch)
           x = F.relu(self.lin1(x))
           x = F.dropout(x, p=0.5, training=self.training)
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
without much modification. Usually, the value of ``batch`` contains two
parts; the first part refers to the result of ``online_transform``,
which generally corresponds to the ``Data`` sequence representing EEG
graphs. The second part refers to the result of ``label_transform``, a
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

Traverse ``k`` folds and train the model separately for testing. It
should be noted that the ``Dataloader`` here needs to use the
implementation in ``PyG`` instead of ``torch``, in order to organize the
``Data`` data structure into ``Batch``.

It is also worth noting that, in general, we need to specify
``shuffle=True`` for the ``DataLoader`` of the training data set to
avoid the deviation of the model training caused by consecutive labels
of the same category.

.. code:: python

   from torch_geometric.loader import DataLoader

   for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
       
       model = GNN().to(device)
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
https://github.com/tczhangzhi/torcheeg/blob/main/examples/examples_torch_geometric.py.
