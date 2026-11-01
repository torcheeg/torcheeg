import importlib
import unittest


class TestVaeExports(unittest.TestCase):
    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.models.vae')
        expected = {'BCDecoder': 'torcheeg.models.vae.bcvae',
                    'BCEncoder': 'torcheeg.models.vae.bcvae',
                    'BDecoder': 'torcheeg.models.vae.bvae',
                    'BEncoder': 'torcheeg.models.vae.bvae'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == "__main__":
    unittest.main()
