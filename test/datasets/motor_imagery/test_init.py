import importlib
import unittest


class TestMotorImageryExports(unittest.TestCase):

    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.datasets.motor_imagery')
        expected = {'BCICIV2aDataset': 'torcheeg.datasets.motor_imagery.bciciv_2a', 'StrokePatientsMIDataset':
                    'torcheeg.datasets.motor_imagery.strokes', 'StrokePatientsMIProcessedDataset': 'torcheeg.datasets.motor_imagery.strokes'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == '__main__':
    unittest.main()
