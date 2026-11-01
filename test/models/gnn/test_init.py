import importlib
import unittest


class TestGnnExports(unittest.TestCase):
    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.models.gnn')
        expected = {'DGCNN': 'torcheeg.models.gnn.dgcnn',
                    'GIN': 'torcheeg.models.gnn.gin',
                    'LGGNet': 'torcheeg.models.gnn.lggnet',
                    'RGNN': 'torcheeg.models.gnn.rgnn'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == "__main__":
    unittest.main()
