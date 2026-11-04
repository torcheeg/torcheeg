import importlib
import unittest


class TestPersonalIdentificationExports(unittest.TestCase):

    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module(
            'torcheeg.datasets.personal_identification')
        expected = {
            'M3CVDataset': 'torcheeg.datasets.personal_identification.m3cv'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == '__main__':
    unittest.main()
