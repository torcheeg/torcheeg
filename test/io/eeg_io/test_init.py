import unittest


class TestPackageExports(unittest.TestCase):
    def test_public_backends_implement_the_package_interface(self):
        import torcheeg.io.eeg_io as module

        for name in ['MemoryEEGIO', 'LMDBEEGIO', 'PickleEEGIO', 'NumpyEEGIO', 'ZarrEEGIO']:
            with self.subTest(name=name):
                self.assertTrue(issubclass(
                    getattr(module, name), module.BaseEEGIO))
