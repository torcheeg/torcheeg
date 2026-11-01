import importlib
import unittest


class TestGenerativeExports(unittest.TestCase):
    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.trainers.generative')
        expected = {'BetaVAETrainer': 'torcheeg.trainers.generative.beta_vae',
                    'CBetaVAETrainer': 'torcheeg.trainers.generative.beta_vae',
                    'CDDPMTrainer': 'torcheeg.trainers.generative.ddpm',
                    'CGlowTrainer': 'torcheeg.trainers.generative.glow',
                    'CWGANGPTrainer': 'torcheeg.trainers.generative.wgan_gp',
                    'DDPMTrainer': 'torcheeg.trainers.generative.ddpm',
                    'GlowTrainer': 'torcheeg.trainers.generative.glow',
                    'WGANGPTrainer': 'torcheeg.trainers.generative.wgan_gp'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == "__main__":
    unittest.main()
