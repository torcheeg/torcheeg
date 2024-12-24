from setuptools import setup, find_packages

__version__ = '1.1.3'
URL = 'https://github.com/tczhangzhi/torcheeg'

install_requires = [
    'tqdm>=4.64.0', 'numpy>=1.21.5', 'pandas>=1.3.5', 'xlrd>=2.0.1',
    'scipy>=1.7.3, <= 1.10.1', 'scikit-learn>=1.0.2', 'lmdb>=1.3.0', 'einops>=0.4.1',
    'mne>=1.0.3', 'xmltodict>=0.13.0', 'networkx>=2.6.3', 'PyWavelets>=1.3.0',
    'spectrum>=0.8.1', 'torchmetrics>=0.10.0', 'mne_connectivity>=0.4.0',
    'pytorch-lightning>=1.9.5', 'wfdb>=4.1.2'
]

test_requires = [
    'torch>=1.10.1',
    'pytest>=7.1.2'
]

model_plugin_requires = [
    'torch-scatter',
    'torch-sparse',
    'torch-cluster',
    'torch-spline-conv',
    'torch_geometric>=2.0.3',
]

trainer_plugin_requires = [
    'torch-fidelity>=0.3.0',
    'torchmetrics[image]>=0.10.0',
]

transforms_plugin_requires = [
    'soxr>=0.3.5',
]

dataset_plugin_requires = [
    'moabb>=0.5.0',
]

dataset_plugin_requires = ['moabb>=0.5.0']

readme = open('README.rst').read()

setup(
    name='torcheeg',
    version='1.1.3',
    description=
    'TorchEEG is a library built on PyTorch for EEG signal analysis. TorchEEG aims to provide a plug-and-play EEG analysis tool, so that researchers can quickly reproduce EEG analysis work and start new EEG analysis research without paying attention to technical details unrelated to the research focus.',
    license='MIT',
    author='TorchEEG Team',
    author_email='tczhangzhi@gmail.com',
    keywords=['PyTorch', 'EEG'],
    url=URL,
    packages=find_packages(),
    long_description=readme,
    python_requires='>=3.7',
    extras_require={
        'test': test_requires,
        'model': model_plugin_requires,
        'dataset': dataset_plugin_requires,
        'trainer': trainer_plugin_requires,
        'transforms': transforms_plugin_requires
    },
    install_requires=install_requires)
