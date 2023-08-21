[pypi-image]: https://badge.fury.io/py/torcheeg.svg
[pypi-url]: https://pypi.python.org/pypi/torcheeg
[docs-image]: https://readthedocs.org/projects/torcheeg/badge/?version=latest
[docs-url]: https://torcheeg.readthedocs.io/en/latest/?badge=latest
[pepy-image]: https://pepy.tech/badge/torcheeg
[pepy-url]: https://pepy.tech/project/torcheeg
[gitter-image]: https://badges.gitter.im/torcheeg/community.svg
[gitter-url]: https://gitter.im/torcheeg/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

![TorchEEG Logo](https://github.com/torcheeg/torcheeg/blob/main/docs/source/_static/torcheeg_logo_dark.png)

--------------------------------------------------------------------------------

[![PyPI Version][pypi-image]][pypi-url]
[![Docs Status][docs-image]][docs-url]
[![Downloads][pepy-image]][pepy-url]
[![Join the Chat][gitter-image]][gitter-url]

**[Documentation](https://torcheeg.readthedocs.io/)** | **[TorchEEG Examples](https://github.com/torcheeg/torcheeg/tree/main/examples)**

TorchEEG is a library built on PyTorch for EEG signal analysis. TorchEEG aims to provide a plug-and-play EEG analysis tool, so that researchers can quickly reproduce EEG analysis work and start new EEG analysis research without paying attention to technical details unrelated to the research focus.

TorchEEG specifies a unified data input-output format (IO) and implement commonly used EEG databases, allowing users to quickly access benchmark datasets and define new custom datasets. The datasets that have been defined so far include emotion recognition and so on. According to papers published in the field of EEG analysis, TorchEEG provides data preprocessing methods commonly used for EEG signals, and provides plug-and-play API for both offline and online pre-proocessing. Offline processing allow users to process once and use any times, speeding up the training process. Online processing allows users to save time when creating new data processing methods. TorchEEG also provides deep learning models following published papers for EEG analysis, including convolutional neural networks, graph convolutional neural networks, and Transformers.

## Installation

TorchEEG depends on PyTorch, please complete the installation of PyTorch according to the system, CUDA version and other information:

```bash
# Conda
# please refer to https://pytorch.org/get-started/locally/
# e.g. CPU version
conda install pytorch==1.11.0 torchvision torchaudio cpuonly -c pytorch
# e.g. GPU version
conda install pytorch==1.11.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Pip
# please refer to https://pytorch.org/get-started/previous-versions/
# e.g. CPU version
pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
# e.g. GPU version
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Anaconda

TorchEEG supports installing with conda! You can simply install TorchEEG using Anaconda, just run the following command:

```bash
conda install -c tczhangzhi -c conda-forge torcheeg
```

### Pip

TorchEEG allows pip-based installation, please use the following command:

```bash
pip install torcheeg
```

### Nightly

In case you want to experiment with the latest TorchEEG features which are not fully released yet, please run the following command to install from the main branch on github:

```bash
pip install git+https://github.com/torcheeg/torcheeg.git
```

### Plugin

TorchEEG provides plugins related to graph algorithms for converting EEG in datasets into graph structures and analyzing them using graph neural networks. This part of the implementation relies on PyG.

> If you do not use graph-related algorithms, you can skip this part of the installation.

```bash
# Conda
# please refer to https://pytorch-geometric.readthedocås.io/en/latest/notes/installation.html
conda install pyg -c pyg

# Pip
# please refer to https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
# e.g. CPU version
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
# e.g. GPU version
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
```

## More About TorchEEG

At a granular level, PyTorch is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| [torcheeg.io](https://torcheeg.readthedocs.io/en/latest/torcheeg.io.html) | A set of unified input and output API is used to store the processing results of various EEG databases for more efficient and convenient use. |
| [torcheeg.datasets](https://torcheeg.readthedocs.io/en/latest/torcheeg.datasets.html) | The packaged benchmark dataset implementation provides a multi-process preprocessing interface. |
| [torcheeg.transforms](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.html) | Extensive EEG preprocessing methods help users extract features, construct EEG signal representations, and connect to commonly used deep learning libraries. |
| [torcheeg.model_selection](https://torcheeg.readthedocs.io/en/latest/torcheeg.model_selection.html) | Extensive dataset partitioning methods for users to experiment with different settings. |
| [torcheeg.models](https://torcheeg.readthedocs.io/en/latest/torcheeg.models.html) | Extensive baseline method reproduction. |

## Implemented Modules

We list currently supported datasets, transforms, data splitting, and deep learning models by category.

**Datasets:** All datasets rely on a set of efficient IO APIs, [torcheeg.io](https://torcheeg.readthedocs.io/en/latest/torcheeg.io.html), to store data preprocessing results on disk and read them quickly during training. Data preprocessing and storage support multiprocessing (speed up!).

* [AMIGOS dataset](https://torcheeg.readthedocs.io/en/latest/torcheeg.datasets.html#amigosdataset) from Miranda-Correa et al.: [AMIGOS: A dataset for affect, personality and mood research on individuals and groups](https://ieeexplore.ieee.org/abstract/document/8554112/).
* [DREAMER dataset](https://torcheeg.readthedocs.io/en/latest/torcheeg.datasets.html#dreamerdataset) from Katsigiannis et al.: [DREAMER: A database for emotion recognition through EEG and ECG signals from wireless low-cost off-the-shelf devices](https://ieeexplore.ieee.org/abstract/document/7887697).
* [SEED dataset](https://torcheeg.readthedocs.io/en/latest/torcheeg.datasets.html#seeddataset) from Zheng et al.: [Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks](https://ieeexplore.ieee.org/abstract/document/7104132).
* [DEAP dataset](https://torcheeg.readthedocs.io/en/latest/torcheeg.datasets.html#deapdataset) from Koelstra et al.: [DEAP: A database for emotion analysis; using physiological signals](https://ieeexplore.ieee.org/abstract/document/5871728).

<details>
<summary><b>Expand to see more...</b></summary>

* [MAHNOB dataset](https://torcheeg.readthedocs.io/en/latest/torcheeg.datasets.html#mahnobdataset) from Soleymani et al.: [A multimodal database for affect recognition and implicit tagging](https://ieeexplore.ieee.org/abstract/document/5975141).
</details>

**Transforms:** TorchEEG provides extensive data transformation tools to help users build EEG data representations suitable for a variety of task formulation and a variety of model structures.

* Feature Engineering: [BandDifferentialEntropy](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-banddifferentialentropy), [BandPowerSpectralDensity](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-bandpowerspectraldensity), [BandMeanAbsoluteDeviation](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-bandmeanabsolutedeviation), [BandKurtosis](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-bandkurtosis), [BandSkewness](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-bandskewness), [Concatenate](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-concatenate)
* General Operation: [PickElectrode](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-pickelectrode), [MeanStdNormalize](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-meanstdnormalize), [MinMaxNormalize](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-minmaxnormalize)
* For CNN: [To2d](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-to2d), [ToGrid](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-togrid), [ToInterpolatedGrid](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.numpy.html#transforms-tointerpolatedgrid)

<details>
<summary><b>Expand to see more...</b></summary>

* For GNN: [ToG](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.pyg.html#transforms-tog)
* For Augmentation: [Resize](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.torch.html#transforms-resize), [RandomNoise](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.torch.html#transforms-randomnoise), [RandomMask](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.torch.html#transforms-randommask)
* For Label Construction: [Select](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.label.html#transforms-select), [Binary](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.label.html#transforms-binary), [BinariesToCategory](https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.label.html#transforms-binariestocategory)
</details>

**Data Splitting:** In current research in the field of EEG analysis, there are various settings based on different considerations for data partitioning. Please choose a reasonable data division method according to the research focus:

* Subject Dependent: [KFoldPerSubjectGroupbyTrial](https://torcheeg.readthedocs.io/en/latest/torcheeg.model_selection.html#kfoldpersubjectgroupbytrial), [train_test_split_per_subject_groupby_trial](https://torcheeg.readthedocs.io/en/latest/torcheeg.model_selection.html#train-test-split-per-subject-groupby-trial)
* Subject Independent: [LeaveOneSubjectOut](https://torcheeg.readthedocs.io/en/latest/torcheeg.model_selection.html#leaveonesubjectout)
* Conventional: [KFold](https://torcheeg.readthedocs.io/en/latest/torcheeg.model_selection.html#kfold), [train_test_split](https://torcheeg.readthedocs.io/en/latest/torcheeg.model_selection.html#train-test-split), [KFoldGroupbyTrial](https://torcheeg.readthedocs.io/en/latest/torcheeg.model_selection.html#kfoldgroupbytrial), [train_test_split_groupby_trial](https://torcheeg.readthedocs.io/en/latest/torcheeg.model_selection.html#train-test-split-groupby-trial)

**Models:** Coming soon after pushing to align with the official implementation or description. If the current version of [CNNs](https://torcheeg.readthedocs.io/en/latest/torcheeg.models.cnn.html), [GNNs](https://torcheeg.readthedocs.io/en/latest/torcheeg.models.gnn.html) and [Transformers](https://torcheeg.readthedocs.io/en/latest/torcheeg.models.transformer.html) is to be used, please refer to the implementation in [torcheeg.models](https://torcheeg.readthedocs.io/en/latest/torcheeg.models.html).

## Quickstart

In this quick tour, we highlight the ease of starting an EEG analysis research with only a few lines.

The `torcheeg.datasets` module contains dataset classes for many real-world EEG datasets. In this tutorial, we use the `DEAP` dataset. We first go to the official website to apply for data download permission according to the introduction of [DEAP dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/), and download the dataset. Next, we need to specify the download location of the dataset in the `root_path` parameter. For the DEAP dataset, we specify the path to the `data_preprocessed_python` folder, e.g. `./tmp_in/data_preprocessed_python`.

```python
from torcheeg.datasets import DEAPDataset
from torcheeg import transforms

from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT

dataset = DEAPDataset(
    io_path=f'./tmp_out/examples_pipeline/deap',
    root_path='./tmp_in/data_preprocessed_python',
    offline_transform=transforms.Compose([
        transforms.BandDifferentialEntropy(apply_to_baseline=True),
        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True)
    ]),
    online_transform=transforms.Compose(
        [transforms.BaselineRemoval(),
         transforms.ToTensor()]),
    label_transform=transforms.Compose([
        transforms.Select('valence'),
        transforms.Binary(5.0),
    ]),
    num_worker=8)
```

The `DEAPDataset` API further contains three parameters: `online_transform`, `offline_transform`, and `label_transform`, which are used to modify samples and labels, respectively.

Here, `offline_transform` will only be called once when the dataset is initialized to preprocess all samples in the dataset, and the processed dataset will be stored in `io_path` to avoid time-consuming repeated transformations in subsequent use. If offline preprocessing is a computationally intensive operation, we also recommend setting multi-CPU parallelism for offline_transform, e.g., set `num_worker` to 4.

`online_transform` is used to transform samples on the fly. Please use `online_transform` if you don't want to wait for the preprocessing of the entire dataset (suitable for scenarios where new `transform` algorithms are designed) or expect data transformation with randomness each time a sample is indexed.

Next, we need to divide the dataset into a training set and a test set. In the field of EEG analysis, commonly used data partitioning methods include k-fold cross-validation and leave-one-out cross-validation. In this tutorial, we use k-fold cross-validation on the entire dataset (`KFold`) as an example of dataset splitting.

```python
from torcheeg.datasets import KFoldGroupbyTrial

k_fold = KFoldGroupbyTrial(n_splits=10,
                           split_path='./tmp_out/examples_pipeline/split',
                           shuffle=True,
                           random_state=42)
```

We loop through each cross-validation set, and for each one, we initialize the CCNN model and define its hyperparameters. For instance, each EEG sample contains 4-channel features from 4 sub-bands, and the grid size is 9x9. We then train the model for 50 epochs using the `ClassifierTrainer`.

```python
from torch.utils.data import DataLoader
from torcheeg.models import CCNN

from torcheeg.trainers import ClassifierTrainer

import pytorch_lightning as pl

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9))

    trainer = ClassifierTrainer(model=model,
                                num_classes=2,
                                lr=1e-4,
                                weight_decay=1e-4,
                                accelerator="gpu")
    trainer.fit(train_loader,
                val_loader,
                max_epochs=50,
                default_root_dir=f'./tmp_out/examples_pipeline/model/{i}',
                callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
                enable_progress_bar=True,
                enable_model_summary=True,
                limit_val_batches=0.0)
    score = trainer.test(val_loader,
                         enable_progress_bar=True,
                         enable_model_summary=True)[0]
    print(f'Fold {i} test accuracy: {score["test_accuracy"]:.4f}')
```

For more specific usage of each module, please refer to [the documentation]((https://torcheeg.readthedocs.io/)).

## Releases and Contributing

TorchEEG is currently in beta; Please let us know if you encounter a bug by filing an issue. We also appreciate all contributions.

If you would like to contribute new datasets, deep learning methods, and extensions to the core, please first open an issue and then send a PR. If you are planning to contribute back bug fixes, please do so without any further discussion.

## Thanks for the support
[![Stargazers repo roster for @torcheeg/torcheeg](https://reporoster.com/stars/torcheeg/torcheeg)](https://github.com/torcheeg/torcheeg/stargazers)

## Thanks for the contributors
<!-- readme: collaborators,contributors -start -->
<table>
<tr>
    <td align="center">
        <a href="https://github.com/tczhangzhi">
            <img src="https://avatars.githubusercontent.com/u/14837418?v=4" width="100;" alt="tczhangzhi"/>
            <br />
            <sub><b>ZHANG Zhi</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/talhaanwarch">
            <img src="https://avatars.githubusercontent.com/u/37379131?v=4" width="100;" alt="talhaanwarch"/>
            <br />
            <sub><b>Talha Anwar</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/MicheleBarsotti">
            <img src="https://avatars.githubusercontent.com/u/48720814?v=4" width="100;" alt="MicheleBarsotti"/>
            <br />
            <sub><b>Meekele_b</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/gitter-badger">
            <img src="https://avatars.githubusercontent.com/u/8518239?v=4" width="100;" alt="gitter-badger"/>
            <br />
            <sub><b>The Gitter Badger</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/arrogant-R">
            <img src="https://avatars.githubusercontent.com/u/83148307?v=4" width="100;" alt="arrogant-R"/>
            <br />
            <sub><b>Null</b></sub>
        </a>
    </td></tr>
</table>
<!-- readme: collaborators,contributors -end -->

## License

TorchEEG has a MIT license, as found in the [LICENSE](https://github.com/torcheeg/torcheeg/blob/main/LICENSE) file.