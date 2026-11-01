import importlib
import unittest


class TestAuditoryAttentionExports(unittest.TestCase):
    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module(
            'torcheeg.datasets.auditory_attention')
        expected = {'AVEDDataset': 'torcheeg.datasets.auditory_attention.aved',
                    'AVEDProcessedDataset': 'torcheeg.datasets.auditory_attention.aved_processed',
                    'DTUDataset': 'torcheeg.datasets.auditory_attention.dtu',
                    'DTUProcessedDataset': 'torcheeg.datasets.auditory_attention.dtu_processed',
                    'KULDataset': 'torcheeg.datasets.auditory_attention.kul',
                    'KULProcessedDataset': 'torcheeg.datasets.auditory_attention.kul_processed'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == "__main__":
    unittest.main()
