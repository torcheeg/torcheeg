import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from torcheeg.models import TFMTokenizer, TFMEncoder


class GeneratedDataset:
    def __init__(self, **kwargs):
        self.info = pd.DataFrame(
            dict(subject_id=['a', 'a', 'b', 'b'], label=[0, 1, 0, 1])
        )

    def __getitem__(self, index):
        return torch.sin(torch.arange(16 * 800).float() / 31).reshape(
            16, 800
        ) * 0.3 + index * 0.1, index % 2


class TestTFMPretrainingExamples(unittest.TestCase):
    def check_complete_generated_workflow(self, objective, quantization_mode='native'):
        name = 'tokenizer' if objective == 'tokenizer' else 'mtp'
        path = (
            Path(__file__).resolve().parents[2]
            / 'examples'
            / ('examples_eegmat_tfm_' + name + '_training.py')
        )
        spec = importlib.util.spec_from_file_location('tfm_' + name + '_example', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'Subject00_1.edf'
            raw.write_bytes(b'Generated reader placeholder; not real EEG.')
            tokenizer_path, encoder_path = root / 'tokenizer.pt', root / 'encoder.pt'
            tokenizer = TFMTokenizer(reconstruction_head='mlp')
            tokenizer.quantizer.cluster_size.fill_(1.0)
            torch.save(tokenizer.state_dict(), tokenizer_path)
            torch.save(TFMEncoder(num_classes=8192).state_dict(), encoder_path)

            def run(name, steps, resume=None):
                args = [
                    'training',
                    '--root-path',
                    directory,
                    '--cache-path',
                    str(root / 'cache'),
                    '--steps',
                    str(steps),
                    '--output-path',
                    str(root / name),
                ]
                if objective == 'tokenizer':
                    args += ['--quantization-mode', quantization_mode]
                if resume:
                    args += ['--resume', str(root / resume)]
                else:
                    args += ['--tokenizer-checkpoint', str(tokenizer_path)]
                    if objective == 'mtp':
                        args += ['--encoder-checkpoint', str(encoder_path)]
                with patch.object(sys, 'argv', args), patch.object(
                    module, 'MentalArithmeticDataset', GeneratedDataset
                ):
                    module.main()

            run('step2.pt', 2)
            run('resumed.pt', 3, 'step2.pt')
            run('continuous.pt', 3)
            resumed = torch.load(root / 'resumed.pt', weights_only=True)
            continuous = torch.load(root / 'continuous.pt', weights_only=True)
            self.assertEqual(
                len(resumed['model']), 192 if objective == 'tokenizer' else 251
            )
            self.assertEqual(resumed['protocol']['train_subjects'], ['a'])
            self.assertEqual(resumed['protocol']['held_out_subject'], 'b')
            self.assertEqual(resumed['protocol']['objective'], objective)
            self.assertEqual(resumed['validation_loss'], continuous['validation_loss'])
            self.assertEqual(resumed['cursor'], continuous['cursor'])
            for name, value in resumed['model'].items():
                torch.testing.assert_close(
                    value, continuous['model'][name], rtol=0, atol=0
                )
            for index, state in resumed['optimizer']['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(
                        value,
                        continuous['optimizer']['state'][index][name],
                        rtol=0,
                        atol=0,
                    )
            for name in ('mask_generator', 'torch_rng'):
                torch.testing.assert_close(
                    resumed[name], continuous[name], rtol=0, atol=0
                )
            if objective == 'mtp':
                for name, value in tokenizer.state_dict().items():
                    torch.testing.assert_close(
                        value, resumed['model']['tokenizer.' + name], rtol=0, atol=0
                    )
            self.assertEqual(list(root.glob('*.tmp')), [])
            raw.write_bytes(b'Changed reader placeholder.')
            with self.assertRaisesRegex(ValueError, 'source files changed'):
                run('changed.pt', 3, 'step2.pt')
            with self.assertRaises(SystemExit):
                run('resumed.pt', 3, 'step2.pt')

    def test_complete_canonical_generated_tokenizer_training_resume(self):
        self.check_complete_generated_workflow('tokenizer')

    def test_complete_canonical_generated_ema_tokenizer_training_resume(self):
        self.check_complete_generated_workflow('tokenizer', 'ema')

    def test_complete_canonical_generated_mtp_training_resume(self):
        self.check_complete_generated_workflow('mtp')


if __name__ == '__main__':
    unittest.main()
