import unittest


class TestPackageExports(unittest.TestCase):
    def test_public_backends_implement_the_package_interface(self):
        import torcheeg.io.info_io as module

        for name in ['MemoryInfoIO', 'CSVInfoIO', 'JSONLInfoIO', 'SQLiteInfoIO']:
            with self.subTest(name=name):
                self.assertTrue(issubclass(
                    getattr(module, name), module.BaseInfoIO))
