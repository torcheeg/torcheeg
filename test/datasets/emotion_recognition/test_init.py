import importlib
import unittest


class TestEmotionRecognitionExports(unittest.TestCase):

    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module(
            'torcheeg.datasets.emotion_recognition')
        expected = {'AMIGOSDataset': 'torcheeg.datasets.emotion_recognition.amigos', 'BCI2022Dataset': 'torcheeg.datasets.emotion_recognition.bci2022', 'DEAPDataset': 'torcheeg.datasets.emotion_recognition.deap', 'DREAMERDataset': 'torcheeg.datasets.emotion_recognition.dreamer', 'FACEDDataset': 'torcheeg.datasets.emotion_recognition.faced', 'FACEDFeatureDataset': 'torcheeg.datasets.emotion_recognition.faced_feature', 'MAHNOBDataset': 'torcheeg.datasets.emotion_recognition.mahnob',
                    'MPEDFeatureDataset': 'torcheeg.datasets.emotion_recognition.mped_feature', 'SEEDDataset': 'torcheeg.datasets.emotion_recognition.seed', 'SEEDFeatureDataset': 'torcheeg.datasets.emotion_recognition.seed_feature', 'SEEDIVDataset': 'torcheeg.datasets.emotion_recognition.seed_iv', 'SEEDIVFeatureDataset': 'torcheeg.datasets.emotion_recognition.seed_iv_feature', 'SEEDVDataset': 'torcheeg.datasets.emotion_recognition.seed_v', 'SEEDVFeatureDataset': 'torcheeg.datasets.emotion_recognition.seed_v_feature'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == '__main__':
    unittest.main()
