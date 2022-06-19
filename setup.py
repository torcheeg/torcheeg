import subprocess
import sys

from pkg_resources import DistributionNotFound, get_distribution
from setuptools.command.install import install
from setuptools import find_packages, setup

__version__ = '1.0.5'
URL = 'http://github.com/tczhangzhi/torcheeg'

pip = "pip3" if sys.version_info[0] == 3 else "pip"


class Install(install):
    def check_torch_geometric_dep(self):
        if self.get_dist('torch') is None:
            self.system(f'{pip} install torch')
        import torch

        if torch.version.cuda is None:
            cuda_version = "+cpu"
        else:
            cuda_version = f"+cu{torch.version.cuda.replace('.', '')}"
        torch_version = torch.__version__.split('.')
        torch_version = '.'.join(torch_version[:-2] + ['0'])
        whl_args = f'-f http://data.pyg.org/whl/torch-{torch_version}{cuda_version}.html --trusted-host data.pyg.org'

        dgl_requires = ['torch-scatter', 'torch-sparse', 'torch-cluster', 'torch-spline-conv']

        for dgl_require in dgl_requires:
            self.system(f'{pip} install {dgl_require} {whl_args}')

    def run(self):
        self.check_torch_geometric_dep()
        install.run(self)

    def system(self, cmd: str):
        subprocess.check_output(cmd, shell=True)

    def get_dist(self, pkgname):
        try:
            return get_distribution(pkgname)
        except DistributionNotFound:
            return None


install_requires = [
    'tqdm', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'lmdb', 'einops', 'torch_geometric', 'mne', 'xmltodict',
    'networkx'
]

test_requires = [
    'pytest',
    'pytest-cov',
]

example_requires = ['pytorch-lightning']

readme = open('README.rst').read()

setup(
    name='torcheeg',
    version='1.0.5',
    description=
    'TorchEEG is a library built on PyTorch for EEG signal analysis. TorchEEG aims to provide a plug-and-play EEG analysis tool, so that researchers can quickly reproduce EEG analysis work and start new EEG analysis research without paying attention to technical details unrelated to the research focus.',
    license='MIT',
    author='TorchEEG Team',
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
    cmdclass={
        'install': Install,
    },
    install_requires=install_requires)
