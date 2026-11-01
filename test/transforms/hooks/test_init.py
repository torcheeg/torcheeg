import importlib
import unittest


class TestHooksExports(unittest.TestCase):

    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.transforms.hooks')
        expected = {'after_hook_linear_dynamical_system': 'torcheeg.transforms.hooks.after_hook', 'after_hook_normalize': 'torcheeg.transforms.hooks.after_hook',
                    'after_hook_running_norm': 'torcheeg.transforms.hooks.after_hook', 'before_hook_normalize': 'torcheeg.transforms.hooks.before_hook'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == '__main__':
    unittest.main()
