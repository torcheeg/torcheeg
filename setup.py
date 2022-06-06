from setuptools import setup, find_packages

__version__ = '1.0.1'
URL = 'https://github.com/tczhanzhi/torcheeg'

install_requires = [
    'tqdm',
    'numpy',
    'pandas',
    'scipy',
    'scikit-learn',
    'lmdb',
    'torch',
    'einops',
    'torch_geometric',
]

test_requires = [
    'pytest',
    'pytest-cov',
]

example_requires = ['pytorch-lightning']

readme = open('README.rst').read()

setup(
    name='torcheeg',
    version='1.0.1',
    description=
    'TorchEEG is a library built on PyTorch for EEG signal analysis. TorchEEG aims to provide a plug-and-play EEG analysis tool, so that researchers can quickly reproduce EEG analysis work and start new EEG analysis research without paying attention to technical details unrelated to the research focus.',
    license='MIT',
    author='Zhi ZHANG',
    author_email='tczhangzhi@gmail.com',
    keywords=['PyTorch', 'EEG'],
    url=URL,
    packages=find_packages(),
    long_description=readme,
    python_requires='>=3.8',
    extras_require={
        'example': example_requires,
        'test': test_requires
    },
    install_requires=install_requires)
