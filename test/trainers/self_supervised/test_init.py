import importlib
import unittest


class TestSelfSupervisedExports(unittest.TestCase):

    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.trainers.self_supervised')
        expected = {'BYOLTrainer': 'torcheeg.trainers.self_supervised.byol',
                    'SimCLRTrainer': 'torcheeg.trainers.self_supervised.sim_clr'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == '__main__':
    unittest.main()
