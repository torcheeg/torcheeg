import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class TestOptionalModelDependencies(unittest.TestCase):

    def _run_without(self, modules, body):
        root = Path(__file__).resolve().parents[3]
        script = "\nimport importlib.abc\nimport sys\nimport torch\nimport torcheeg\n\nassert torcheeg.__file__.startswith(ROOT)\n\nblocked = MODULES\n\nclass BlockOptionalDependencies(importlib.abc.MetaPathFinder):\n    def find_spec(self, fullname, path=None, target=None):\n        if fullname.split('.')[0] in blocked:\n            raise ModuleNotFoundError('Blocked optional dependency: ' + fullname)\n\nassert not set(blocked).intersection(sys.modules)\nsys.meta_path.insert(0, BlockOptionalDependencies())\n".replace(
            'MODULES', repr(modules)).replace('ROOT', repr(str(root))) + textwrap.dedent(body)
        env = dict(os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS='1')
        result = subprocess.run([sys.executable, '-c', script], cwd=root,
                                env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_moabb_reports_its_own_install_hint(self):
        self._run_without(('moabb',), "\n            try:\n                import torcheeg.datasets.moabb\n            except ImportError as error:\n                assert 'pip install moabb' in str(error), str(error)\n                assert 'pytorch-geometric' not in str(error), str(error)\n                assert isinstance(error.__cause__, ModuleNotFoundError)\n            else:\n                raise AssertionError('MOABB type imports must report their missing dependency')\n        ")


if __name__ == '__main__':
    unittest.main()
