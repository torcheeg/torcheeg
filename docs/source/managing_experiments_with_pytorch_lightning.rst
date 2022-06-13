Managing Experiments with PyTorch Lightning
===========================================

In this quick tour, we'll take a closer look at how to bring together
TorchEEG and `PyTorch
Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`__ to
organize and run experiments using multiple GPUs.

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

A simple case implemented according to the official documentation of
`pytorch-lightning <https://github.com/PyTorchLightning/pytorch-lightning>`__.
Here, ``__init__``, ``forward``, ``training_step``, ``validation_step``,
and ``configure_optimizers`` need to be implemented, where ``__init__``
is used to specify hyperparameters and initialize related dependencies,
``forward`` is used to define the forward propagation process of the
network, ``training_step`` and ``validation_step`` are used to define
the training and testing process. Usually, the value of ``batch``
contains two parts; the first part refers to the result of
``online_transform``, which generally corresponds to the ``Tensor``
sequence representing EEG signals. The second part refers to the result
of ``label_transform``, a sequence of integers representing the label.
Besides, ``configure_optimizers`` is used to define the required
optimizers and schedulers.

.. code:: python

   class EEGClassifier(LightningModule):
       def __init__(self, model, lr=1e-4):
           super().__init__()
           self.save_hyperparameters(ignore="model")
           self.model = model
           self.val_acc = Accuracy()

       def forward(self, x):
           return self.model(x)

       def training_step(self, batch, batch_idx):
           X = batch[0]
           y = batch[1]

           logits = self.forward(X)
           loss = F.cross_entropy(logits, y.long())
           return loss

       def validation_step(self, batch, batch_idx):
           X = batch[0]
           y = batch[1]

           logits = self.forward(X)
           loss = F.cross_entropy(logits, y.long())

           self.val_acc(logits, y)
           self.log("val_acc", self.val_acc)
           self.log("val_loss", loss)

       def configure_optimizers(self):
           optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

           return [optimizer], []

Traverse ``k`` folds and train the model separately for testing. It is
worth noting that, in general, we need to specify ``shuffle=True`` for
the ``DataLoader`` of the training data set to avoid the deviation of
the model training caused by consecutive labels of the same category.

.. code:: python

   for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
           train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
           val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
           tb_logger = TensorBoardLogger(save_dir='lightning_logs', name=f'fold_{i + 1}')
           checkpoint_callback = ModelCheckpoint(dirpath=tb_logger.log_dir,
                                                 filename="{epoch:02d}-{val_metric:.4f}",
                                                 monitor='val_metric',
                                                 mode='max')

           model = EEGClassifier(CNN())

           trainer = Trainer(max_epochs=50,
                             devices=2,
                             accelerator="auto",
                             strategy="ddp",
                             checkpoint_callback=checkpoint_callback,
                             logger=tb_logger)

           trainer.fit(model, train_loader, val_loader)

For multi-GPU parallel training, just define the number of GPUs using
``device`` and set ``strategy="ddp"``. For full code, please refer to
https://github.com/tczhangzhi/torcheeg/blob/main/examples/examples_torch_lightening.py.
