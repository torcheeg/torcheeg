"""Optional complete separately supplied author models and official checkpoints."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import unittest

import torch
from torch.nn import functional as F

from torcheeg.models import SleepFM, SleepFMStaging, SleepFMDiagnosis


@unittest.skipUnless(
    os.environ.get('TORCHEEG_SLEEPFM_BACKEND_ROOT'),
    'An explicit separately licensed author checkout with weights is required.',
)
class TestSleepFMNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['TORCHEEG_SLEEPFM_BACKEND_ROOT']).resolve()
        revision = subprocess.run(
            ['git', '-C', str(cls.root), 'rev-parse', 'HEAD'],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != '2bcbae04c3592f61352addb7ac3d4193f0a3ca25':
            raise ValueError('Require the documented pinned complete author release.')
        subprocess.run(
            ['git', '-C', str(cls.root), 'diff', '--quiet', 'HEAD'], check=True
        )
        spec = importlib.util.spec_from_file_location(
            'sleepfm_explicit_native', cls.root / 'sleepfm/models/models.py'
        )
        cls.author = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.author)

    def compare(self, variant):
        folder = self.root / 'sleepfm/checkpoints' / ('model_' + variant)
        config = json.loads((folder / 'config.json').read_text())
        checkpoint = next(folder.glob('best.*'))
        if variant == 'base':
            native = self.author.SetTransformer(
                **{
                    k: config[k]
                    for k in [
                        'in_channels',
                        'patch_size',
                        'embed_dim',
                        'num_heads',
                        'num_layers',
                        'pooling_head',
                        'dropout',
                    ]
                }
            )
            model = SleepFM.from_pretrained(checkpoint, native, ['EEG1', 'EEG2'])
        else:
            native = getattr(self.author, config['model'])(**config['model_params'])
            model = (
                SleepFMStaging.from_pretrained(checkpoint, native, ['BAS', 'RESP'])
                if variant == 'sleep_staging'
                else SleepFMDiagnosis.from_pretrained(
                    checkpoint,
                    native,
                    ['BAS', 'RESP'],
                    ['native_column0', 'native_column1'],
                )
            )
        count, elements = {
            'base': (155, 4848550),
            'sleep_staging': (37, 2335365),
            'diagnosis': (35, 1792937),
        }[variant]
        self.assertEqual(len(native.state_dict()), count)
        self.assertEqual(sum(v.numel() for v in native.state_dict().values()), elements)
        reference = copy.deepcopy(native)
        for training in [False, True]:
            model.train(training)
            reference.train(training)
            torch.manual_seed(24)
            values = (
                torch.randn(2, 2, 1280)
                if variant == 'base'
                else torch.randn(2, 2, 4, 128)
            )
            left = values.clone().requires_grad_()
            right = values.clone().requires_grad_()
            mask = torch.zeros(
                values.shape[:2] if variant == 'base' else values.shape[:3],
                dtype=torch.bool,
            )
            mask[1, 1] = True
            if variant != 'base':
                mask[1, :, -1] = True
            demographics = torch.tensor([[0.3, 0.0], [0.4, 1.0]], requires_grad=True)
            reference_demographics = demographics.detach().clone().requires_grad_()
            native.zero_grad()
            reference.zero_grad()
            torch.manual_seed(33)
            if variant == 'base':
                actual = (model(left, mask), model(left, mask, return_tokens=True))
            elif variant == 'sleep_staging':
                actual = (model(left, mask)[0],)
            else:
                actual = (model(left, demographics, mask),)
            torch.manual_seed(33)
            if variant == 'base':
                expected = (reference(right, mask)[0], reference(right, mask)[1])
            elif variant == 'sleep_staging':
                expected = (
                    torch.cat(
                        [
                            F.pad(
                                reference(
                                    right[i : i + 1, :, :n], mask[i : i + 1, :, :n]
                                )[0],
                                (0, 0, 0, 4 - n),
                            )
                            for i, n in enumerate([4, 3])
                        ]
                    ),
                )
            else:
                expected = (reference(right, mask, reference_demographics),)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            sum(a.square().mean() for a in actual).backward()
            sum(b.square().mean() for b in expected).backward()
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(left.grad).all())
            for (name, a), (_, b) in zip(
                native.named_parameters(), reference.named_parameters()
            ):
                self.assertEqual(a.grad is None, b.grad is None, name)
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)
            if variant == 'diagnosis':
                torch.testing.assert_close(
                    demographics.grad, reference_demographics.grad, rtol=0, atol=0
                )
            if variant == 'sleep_staging':
                self.assertEqual(torch.count_nonzero(left.grad[1, :, -1]), 0)
        if variant == 'sleep_staging':
            model.eval()
            with torch.no_grad():
                before = model(values, mask)[0]
                changed = values.clone()
                changed[1, :, -1] += 200
                torch.testing.assert_close(
                    model(changed, mask)[0], before, rtol=0, atol=0
                )
                self.assertEqual(torch.count_nonzero(before[1, -1]), 0)

    def test_complete_upstream(self):
        self.compare('base')

    def test_complete_sleep_staging(self):
        self.compare('sleep_staging')

    def test_complete_diagnosis(self):
        self.compare('diagnosis')


if __name__ == '__main__':
    unittest.main()
