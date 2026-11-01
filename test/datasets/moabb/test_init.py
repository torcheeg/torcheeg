import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class TestOptionalModelDependencies(unittest.TestCase):
    def _run_without(self, modules, body):
        root = Path(__file__).resolve().parents[3]
        script = '''
import importlib.abc
import sys
import torch
import torcheeg

assert torcheeg.__file__.startswith(ROOT)

blocked = MODULES

class BlockOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('Blocked optional dependency: ' + fullname)

assert not set(blocked).intersection(sys.modules)
sys.meta_path.insert(0, BlockOptionalDependencies())
'''.replace('MODULES', repr(modules)).replace('ROOT', repr(str(root))) + textwrap.dedent(body)
        env = dict(os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS='1')
        result = subprocess.run([sys.executable, '-c', script], cwd=root,
                                env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_moabb_reports_its_own_install_hint(self):
        self._run_without(('moabb',), """
            try:
                import torcheeg.datasets.moabb
            except ImportError as error:
                assert 'pip install moabb' in str(error), str(error)
                assert 'pytorch-geometric' not in str(error), str(error)
                assert isinstance(error.__cause__, ModuleNotFoundError)
            else:
                raise AssertionError('MOABB type imports must report their missing dependency')
        """)


if __name__ == '__main__':
    unittest.main()
