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


## Python naming and base classes

Follow the [Google Python Style Guide naming conventions](https://google.github.io/styleguide/pyguide.html#s3.16-naming), with the following rules for TorchEEG:

| Kind | Public API or subclass hook | Internal implementation |
| --- | --- | --- |
| Class | `ClassName` | `_ClassName` |
| Function or method | `method_name` | `_method_name` |
| Instance or module variable | `variable_name` | `_variable_name` |
| Module or class constant | `CONSTANT_NAME` | `_CONSTANT_NAME` |

Use public names for user-facing operations, constructor parameters and their exposed attributes, and methods intended for subclasses to override (for example, `apply`, `set_records`, `read_record`, `process_record`, `build`, and `split`). Use a single leading underscore for implementation helpers and internal state. Keep local variables and function parameters in `snake_case` without a leading underscore, except for intentionally unused values.

Do not introduce double-leading-underscore names for private methods or attributes. Keep Python protocol methods such as `__init__` and `__repr__`, and names required by PyTorch, Lightning, or another upstream interface. Model submodule, parameter, and buffer names form part of the `state_dict` checkpoint interface; changing their spelling requires a checkpoint migration. Public imports and documented extension points also require a compatibility plan before renaming. Existing helpers exported through wildcard imports remain available until such a migration is made.

A base-class method that requires a subclass implementation raises `NotImplementedError` and identifies the missing method and concrete class:

```python
def build(self, **kwargs):
    raise NotImplementedError(
        "Method build is not implemented in class "
        + self.__class__.__name__
    )
```

For a class-level hook, use `@classmethod` and `cls.__name__` in the message. Subclasses can still implement dataset record hooks as static methods. Keep working default implementations in base classes; returning an empty parameter dictionary or implementing `__repr__` does not require an exception. Exceptions for unsupported algorithm options are separate from required subclass hooks.

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