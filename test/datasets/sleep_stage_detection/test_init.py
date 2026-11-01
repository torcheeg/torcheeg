import importlib
import unittest


class TestSleepStageDetectionExports(unittest.TestCase):
    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module(
            'torcheeg.datasets.sleep_stage_detection')
        expected = {'HMCDataset': 'torcheeg.datasets.sleep_stage_detection.hmc',
                    'ISRUCDataset': 'torcheeg.datasets.sleep_stage_detection.isruc',
                    'P2018Dataset': 'torcheeg.datasets.sleep_stage_detection.p2018',
                    'SleepEDFxDataset': 'torcheeg.datasets.sleep_stage_detection.sleep_edfx'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == "__main__":
    unittest.main()
