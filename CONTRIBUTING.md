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
pip install -e ".[test,example,pyg]"
```


## Unit Testing

The TorchEEG unit tests are located under `test/`. Run the all the unit tests with:

```bash
# please skip test/datasets if you don't have dataset folder in tmp_in/
python -m unittest discover test/
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

cd ..
rm -rf src # remove pytorch_sphinx_theme
```

In general, the documentation is available to view by opening `127.0.0.1:8000`.