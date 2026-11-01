import importlib
import unittest


class TestGanExports(unittest.TestCase):
    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.models.gan')
        expected = {'BCDiscriminator': 'torcheeg.models.gan.bcgan',
                    'BCGenerator': 'torcheeg.models.gan.bcgan',
                    'BDiscriminator': 'torcheeg.models.gan.bgan',
                    'BGenerator': 'torcheeg.models.gan.bgan',
                    'EEGfuseNet': 'torcheeg.models.gan.eegfusenet',
                    'EFDiscriminator': 'torcheeg.models.gan.eegfusenet'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == "__main__":
    unittest.main()
