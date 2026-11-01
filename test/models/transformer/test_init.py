import importlib
import unittest


class TestTransformerExports(unittest.TestCase):
    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.models.transformer')
        expected = {'ATCNet': 'torcheeg.models.transformer.atcnet',
                    'ArjunViT': 'torcheeg.models.transformer.arjun_vit',
                    'Conformer': 'torcheeg.models.transformer.conformer',
                    'DARNet': 'torcheeg.models.transformer.darnet',
                    'LaBraM': 'torcheeg.models.transformer.labram',
                    'MHANet': 'torcheeg.models.transformer.mhanet',
                    'SimpleViT': 'torcheeg.models.transformer.simple_vit',
                    'TCNet': 'torcheeg.models.transformer.tcnet',
                    'VanillaTransformer': 'torcheeg.models.transformer.vanilla_transformer',
                    'ViT': 'torcheeg.models.transformer.vit'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == "__main__":
    unittest.main()
