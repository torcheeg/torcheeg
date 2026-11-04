import unittest


class TestPackageExports(unittest.TestCase):
    def test_public_backends_implement_the_package_interface(self):
        import torcheeg.runners as module

        for name in ['SequentialRunner', 'MultiprocessRunner', 'JoblibRunner']:
            with self.subTest(name=name):
                self.assertTrue(issubclass(
                    getattr(module, name), module.BaseRunner))
