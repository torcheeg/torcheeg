import importlib
import unittest


class TestAnyExports(unittest.TestCase):
    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.transforms.any')
        expected = {'BaseTransform': 'torcheeg.transforms.base_transform',
                    'BaselineCorrection': 'torcheeg.transforms.any.baseline',
                    'BaselineRemoval': 'torcheeg.transforms.any.baseline',
                    'Compose': 'torcheeg.transforms.any.compose',
                    'EEGTransform': 'torcheeg.transforms.base_transform',
                    'Lambda': 'torcheeg.transforms.any.lambd',
                    'PDB': 'torcheeg.transforms.any.pdb'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == "__main__":
    unittest.main()
