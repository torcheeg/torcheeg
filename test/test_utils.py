import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from torcheeg.utils import LazyLoader


class TestLazyLoader(unittest.TestCase):
    def test_import_is_deferred_and_cached(self):
        module = SimpleNamespace(value=42)
        with patch('torcheeg.utils._import_module', return_value=module) as load:
            lazy = LazyLoader('graph.backend')
            load.assert_not_called()
            self.assertEqual(lazy.value, 42)
            self.assertEqual(lazy.value, 42)
            load.assert_called_once_with('graph.backend')

    def test_missing_dependency_preserves_cause(self):
        error = ModuleNotFoundError('graph backend unavailable')
        with patch('torcheeg.utils._import_module', side_effect=error):
            lazy = LazyLoader('torch_scatter')
            with self.assertRaisesRegex(ImportError, 'Could not import optional dependency `torch_scatter`') as caught:
                lazy.scatter_add
            self.assertIs(caught.exception.__cause__, error)
            self.assertNotIn("pytorch-geometric", str(caught.exception))

    def test_custom_install_hint_and_retry(self):
        error = ImportError('backend binary incompatible')
        module = SimpleNamespace(value=42)
        with patch('torcheeg.utils._import_module', side_effect=[error, module]):
            lazy = LazyLoader(
                'backend', install_hint='See backend installation docs.')
            with self.assertRaisesRegex(ImportError, 'See backend installation docs') as caught:
                lazy.value
            self.assertIs(caught.exception.__cause__, error)
            self.assertEqual(lazy.value, 42)

    def test_class_attribute_can_be_used_as_a_base(self):
        lazy = LazyLoader('collections')

        class Derived(lazy.UserList):
            pass

        data = Derived([1, 2])
        data.append(3)
        self.assertEqual(data, [1, 2, 3])


class TestUtilsImports(unittest.TestCase):
    def run_fresh(self, body):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, '-c', textwrap.dedent(body)], cwd=root,
            env=dict(os.environ, PYTHONPATH=str(root), MPLBACKEND='Agg'),
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_shared_utilities_do_not_load_visualizers(self):
        self.run_fresh('''
            import importlib.abc
            import sys
            import torcheeg

            class BlockVisualizers(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.startswith('torcheeg.visualizers'):
                        raise AssertionError('Unexpected visualization import: ' + fullname)

            sys.meta_path.insert(0, BlockVisualizers())
            import torcheeg.utils as utils
            from torcheeg.utils import download_url, get_random_dir_path, LazyLoader
            assert download_url is utils.download_url
            assert not any(name.startswith('torcheeg.visualizers') for name in sys.modules)
        ''')


if __name__ == '__main__':
    unittest.main()
