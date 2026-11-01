import importlib
import unittest


class TestSsvepExports(unittest.TestCase):
    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.datasets.ssvep')
        expected = {'SanDiegoSSVEPDataset': 'torcheeg.datasets.ssvep.sandiego_ssvep',
                    'TSUBenckmarkDataset': 'torcheeg.datasets.ssvep.tsu_benchmark'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == "__main__":
    unittest.main()
