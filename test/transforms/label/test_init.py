import importlib
import unittest


class TestLabelExports(unittest.TestCase):
    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.transforms.label')
        expected = {'BinariesToCategory': 'torcheeg.transforms.label.binary',
                    'Binary': 'torcheeg.transforms.label.binary',
                    'BinaryOneVSRest': 'torcheeg.transforms.label.binary',
                    'FixCategory': 'torcheeg.transforms.label.fix',
                    'LabelTransform': 'torcheeg.transforms.base_transform',
                    'Mapping': 'torcheeg.transforms.label.mapping',
                    'Multilabel': 'torcheeg.transforms.label.multilabel',
                    'Normalize': 'torcheeg.transforms.label.normalize',
                    'Select': 'torcheeg.transforms.label.select',
                    'StringToInt': 'torcheeg.transforms.label.string'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == "__main__":
    unittest.main()
