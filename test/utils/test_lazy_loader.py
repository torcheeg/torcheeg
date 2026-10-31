import unittest
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
            lazy = LazyLoader('backend', install_hint='See backend installation docs.')
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


if __name__ == '__main__':
    unittest.main()
