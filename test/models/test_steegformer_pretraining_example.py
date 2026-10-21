import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from torcheeg.models import STEEGFormerPretrainer


class TinyDataset:
    def __init__(self, **kwargs):
        self.info = pd.DataFrame(
            dict(subject_id=['a', 'a', 'b', 'b'], label=[0, 1, 0, 1])
        )

    def __getitem__(self, index):
        return torch.linspace(-1.0, 1.0, 19 * 16).reshape(
            19, 16
        ) + index * 0.1, index % 2


class TestSTEEGFormerPretrainingExample(unittest.TestCase):
    def test_standalone_cli_complete_resume_matches_continuous_and_rejects_changed_sources(
        self,
    ):
        path = (
            Path(__file__).resolve().parents[2]
            / 'examples'
            / 'examples_eegmat_steegformer_pretraining.py'
        )
        spec = importlib.util.spec_from_file_location('steegformer_mae_example', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = dict(
            patch_size=4,
            hid_channels=8,
            mlp_channels=32,
            depth=1,
            num_heads=2,
            decoder_channels=8,
            decoder_depth=1,
            decoder_num_heads=2,
        )
        original_loader = STEEGFormerPretrainer.from_pretrained

        def loader(path, variant=None, **kwargs):
            return original_loader(path, **config, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'Subject00_1.edf'
            raw.write_bytes(b'Independent mocked reader input, not a real EDF.')
            checkpoint = root / 'source.pt'
            torch.save(STEEGFormerPretrainer(**config).state_dict(), checkpoint)

            def run(name, steps, resume=None):
                argv = [
                    'pretraining',
                    '--root-path',
                    directory,
                    '--cache-path',
                    str(root / 'cache'),
                    '--steps',
                    str(steps),
                    '--output-path',
                    str(root / name),
                ]
                argv += (
                    ['--resume', str(root / resume)]
                    if resume
                    else ['--checkpoint', str(checkpoint)]
                )
                with patch.object(sys, 'argv', argv), patch.object(
                    module, 'MentalArithmeticDataset', TinyDataset
                ), patch.object(
                    module.STEEGFormerPretrainer, 'from_pretrained', side_effect=loader
                ):
                    module.main()

            run('step2.pt', 2)
            run('resumed.pt', 3, 'step2.pt')
            run('continuous.pt', 3)
            resumed = torch.load(root / 'resumed.pt', weights_only=True)
            continuous = torch.load(root / 'continuous.pt', weights_only=True)
            self.assertEqual(resumed['protocol']['train_subjects'], ['a'])
            self.assertEqual(resumed['protocol']['held_out_subject'], 'b')
            self.assertEqual(resumed['validation_loss'], continuous['validation_loss'])
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
            torch.testing.assert_close(
                resumed['mask_generator'], continuous['mask_generator'], rtol=0, atol=0
            )
            self.assertEqual(list(root.glob('*.tmp')), [])
            with self.assertRaises(SystemExit):
                run('resumed.pt', 3, 'step2.pt')
            raw.write_bytes(b'Changed mocked reader input.')
            with self.assertRaisesRegex(ValueError, 'original recordings changed'):
                run('changed.pt', 3, 'step2.pt')


if __name__ == '__main__':
    unittest.main()
