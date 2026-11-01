import importlib
import unittest


class TestImbalanceExports(unittest.TestCase):
    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.trainers.imbalance')
        expected = {'EQLossTrainer': 'torcheeg.trainers.imbalance.eq',
                    'FocalLossTrainer': 'torcheeg.trainers.imbalance.focal',
                    'LALossTrainer': 'torcheeg.trainers.imbalance.la',
                    'LDAMLossTrainer': 'torcheeg.trainers.imbalance.ldam',
                    'WCELossTrainer': 'torcheeg.trainers.imbalance.wce'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == "__main__":
    unittest.main()
