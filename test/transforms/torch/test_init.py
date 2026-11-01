import importlib
import unittest


class TestTorchExports(unittest.TestCase):

    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.transforms.torch')
        expected = {'Contrastive': 'torcheeg.transforms.torch.contrastive', 'EEGTransform': 'torcheeg.transforms.base_transform', 'RandomChannelShuffle': 'torcheeg.transforms.torch.random', 'RandomEEGTransform': 'torcheeg.transforms.torch.random', 'RandomFlip': 'torcheeg.transforms.torch.random', 'RandomFrequencyShift': 'torcheeg.transforms.torch.random', 'RandomHemisphereChannelShuffle': 'torcheeg.transforms.torch.random', 'RandomMask': 'torcheeg.transforms.torch.random',
                    'RandomMask2D': 'torcheeg.transforms.torch.random', 'RandomMaskGrid': 'torcheeg.transforms.torch.random', 'RandomNoise': 'torcheeg.transforms.torch.random', 'RandomPCANoise': 'torcheeg.transforms.torch.random', 'RandomShift': 'torcheeg.transforms.torch.random', 'RandomSignFlip': 'torcheeg.transforms.torch.random', 'RandomWindowSlice': 'torcheeg.transforms.torch.random', 'RandomWindowWarp': 'torcheeg.transforms.torch.random', 'Resize': 'torcheeg.transforms.torch.resize', 'ToTensor': 'torcheeg.transforms.torch.to'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == '__main__':
    unittest.main()
