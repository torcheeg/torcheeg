import importlib
import unittest


class TestDomainAdaptionExports(unittest.TestCase):
    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.trainers.domain_adaption')
        expected = {'ADATrainer': 'torcheeg.trainers.domain_adaption.ada',
                    'CORALTrainer': 'torcheeg.trainers.domain_adaption.coral',
                    'CenterLossTrainer': 'torcheeg.trainers.domain_adaption.center',
                    'DANNTrainer': 'torcheeg.trainers.domain_adaption.dann',
                    'DANTrainer': 'torcheeg.trainers.domain_adaption.dan',
                    'DDCTrainer': 'torcheeg.trainers.domain_adaption.ddc',
                    'JANTrainer': 'torcheeg.trainers.domain_adaption.jan'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == "__main__":
    unittest.main()
