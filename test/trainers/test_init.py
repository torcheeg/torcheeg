import importlib
import unittest


class TestTrainersExports(unittest.TestCase):
    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.trainers')
        expected = {'ADATrainer': 'torcheeg.trainers.domain_adaption.ada',
                    'BYOLTrainer': 'torcheeg.trainers.self_supervised.byol',
                    'BetaVAETrainer': 'torcheeg.trainers.generative.beta_vae',
                    'CBetaVAETrainer': 'torcheeg.trainers.generative.beta_vae',
                    'CDDPMTrainer': 'torcheeg.trainers.generative.ddpm',
                    'CGlowTrainer': 'torcheeg.trainers.generative.glow',
                    'CORALTrainer': 'torcheeg.trainers.domain_adaption.coral',
                    'CWGANGPTrainer': 'torcheeg.trainers.generative.wgan_gp',
                    'CenterLossTrainer': 'torcheeg.trainers.domain_adaption.center',
                    'ClassifierTrainer': 'torcheeg.trainers.classifier',
                    'DANNTrainer': 'torcheeg.trainers.domain_adaption.dann',
                    'DANTrainer': 'torcheeg.trainers.domain_adaption.dan',
                    'DDCTrainer': 'torcheeg.trainers.domain_adaption.ddc',
                    'DDPMTrainer': 'torcheeg.trainers.generative.ddpm',
                    'EQLossTrainer': 'torcheeg.trainers.imbalance.eq',
                    'FocalLossTrainer': 'torcheeg.trainers.imbalance.focal',
                    'GlowTrainer': 'torcheeg.trainers.generative.glow',
                    'JANTrainer': 'torcheeg.trainers.domain_adaption.jan',
                    'LALossTrainer': 'torcheeg.trainers.imbalance.la',
                    'LDAMLossTrainer': 'torcheeg.trainers.imbalance.ldam',
                    'RegressorTrainer': 'torcheeg.trainers.regressor',
                    'RootMeanSquaredError': 'torcheeg.trainers.regressor',
                    'SimCLRTrainer': 'torcheeg.trainers.self_supervised.sim_clr',
                    'WCELossTrainer': 'torcheeg.trainers.imbalance.wce',
                    'WGANGPTrainer': 'torcheeg.trainers.generative.wgan_gp',
                    'classification_metrics': 'torcheeg.trainers.classifier',
                    'regression_metrics': 'torcheeg.trainers.regressor'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == "__main__":
    unittest.main()
