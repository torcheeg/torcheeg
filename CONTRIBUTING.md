# Contributing to TorchEEG

If you are interested in contributing to TorchEEG, your contribution may fall into one of two categories:

1. You want to implement a new feature:
   
     We are happy to accept the implementation of any algorithms, datasets, experiments in the field of EEG analysis. If you need help with feature design/implementation, please post in an issue.
2. You want to fix a bug:
   
     Feel free to send a pull request when you encounter a bug. We recommend that you describe the bug you encounter before submitting your code. If you're not sure if this is a bug or how to fix it, please post in an issue.

Please send a Pull Request to https://github.com/tczhangzhi/torcheeg!

## Developing TorchEEG

To develop TorchEEG, here are some tips:

Uninstall existing TorchEEG on your machine:

```bash
pip uninstall torcheeg
```

Clone the source code of TorchEEG:

```bash
git clone https://github.com/tczhangzhi/torcheeg
cd torcheeg
```

Install dependencies required during development:

```bash
python -m pip install -e ".[test,model,trainer,transforms,dataset]"
```


Use a dedicated environment. Install matching PyTorch and torchvision versions
from the same CPU/CUDA build source using the
[official installation instructions](https://pytorch.org/get-started/previous-versions/).
Install graph extension wheels matching that PyTorch version and build using the
[PyG instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
before installing the development extras above. The full suite includes graph
tests, so it requires those plugins even when your own model does not.
See [.github/workflows/test.yml](.github/workflows/test.yml) for the exact
Python 3.8 / PyTorch 2.4.1 CPU environment used by CI.

Before running the full suite, check dependency consistency and imports:

```bash
python -m pip check
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
python -c "import torch_geometric, torch_scatter, torch_sparse"
```

If discovery reports missing `torch_scatter` or `torch_geometric`, install
matching graph plugins in that environment. The current RGNN module also imports
`torch_scatter` when the model package is imported, so one missing plugin can
cause failures in several non-graph test modules.
[PR #228](https://github.com/torcheeg/torcheeg/pull/228) separately addresses
that eager import.

If torchvision reports `operator torchvision::nms does not exist`, check the
torch/torchvision version pair and binary build compatibility. Recreate the
dedicated environment with matching official wheels before rerunning discovery.
Do not hide the import failure by skipping trainer tests. A successful
`pip check` alone does not verify that compiled operators load.

## Unit Testing

The TorchEEG unit tests are located under `test/`. Run the all the unit tests with:

```bash
# please skip test/datasets if you don't have dataset folder in tmp_in/
python -m unittest discover -s test -t .
# run IO tests by their package names
python -m unittest test.io.test_eeg_signal test.io.test_meta_info
# save to file
python -m unittest discover -s test -t . > log.txt 2>&1
```

or test individual files via:

```bash
export PYTHONPATH=./
# for example, test_numpy.py
python test/transforms/test_numpy.py
```

## Building

For local tests for pip installs, please run:

```bash
pip install .
```

If everything is as expected, keep running the following lines to build the package and upload it to PyPI:

```bash
python setup.py sdist
twine upload dist/torcheeg-x.x.x.tar.gz
```

For local testing of conda installations, please run:

```bash
conda build . -c conda-forge
conda install --use-local PATH/TO/torcheeg-x.x.x-0.tar.bz2
```

If everything is as expected, keep running the following lines to build the package and upload it to anaconda:

```bash
conda install anaconda-client
anaconda login
anaconda upload PATH/TO/torcheeg-x.x.x-0.tar.bz2
```

To build the documentation, please run:

```bash
pip install -e git+https://github.com/pytorch/pytorch_sphinx_theme.git#egg=pytorch_sphinx_theme

cd docs

make clean
sphinx-autobuild source build/html

# or
make html
cd build/html
python -m http.server 8000

cd ..
rm -rf src # remove pytorch_sphinx_theme
```

In general, the documentation is available to view by opening `127.0.0.1:8000`.