import importlib
import unittest


class TestCnnExports(unittest.TestCase):
    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.models.cnn')
        expected = {'CCNN': 'torcheeg.models.cnn.ccnn',
                    'CSPNet': 'torcheeg.models.cnn.cspnet',
                    'DeepSleepNet': 'torcheeg.models.cnn.deep_sleepnet',
                    'EEGNet': 'torcheeg.models.cnn.eegnet',
                    'FBCCNN': 'torcheeg.models.cnn.fbccnn',
                    'FBCNet': 'torcheeg.models.cnn.fbcnet',
                    'FBMSNet': 'torcheeg.models.cnn.fbmsnet',
                    'LMDA': 'torcheeg.models.cnn.lmda',
                    'MTCNN': 'torcheeg.models.cnn.mtcnn',
                    'SSTEmotionNet': 'torcheeg.models.cnn.sst_emotion_net',
                    'STNet': 'torcheeg.models.cnn.stnet',
                    'TSCeption': 'torcheeg.models.cnn.tsception',
                    'TSLANet': 'torcheeg.models.cnn.tslanet',
                    'TinySleepNet': 'torcheeg.models.cnn.tiny_sleepnet',
                    'USleep': 'torcheeg.models.cnn.usleep'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == "__main__":
    unittest.main()
