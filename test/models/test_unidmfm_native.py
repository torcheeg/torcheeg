"""Optional oracle against complete, separately obtained Uni-DMFM source/weights."""

import copy
import importlib.util
import os
from pathlib import Path
import unittest

import torch

from torcheeg.models import UniDMFM


@unittest.skipUnless(
    os.environ.get('TORCHEEG_UNIDMFM_BACKEND_ROOT')
    and os.environ.get('TORCHEEG_UNIDMFM_CHECKPOINT'),
    'Explicit trusted source and complete official checkpoint required.',
)
class TestUniDMFMNative(unittest.TestCase):
    def test_complete_original_all_outputs_and_every_gradient(self):
        script = (
            Path(__file__).resolve().parents[2] / 'examples/examples_eegmat_unidmfm.py'
        )
        spec = importlib.util.spec_from_file_location(
            'unidmfm_explicit_example', script
        )
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        root, checkpoint = (
            os.environ['TORCHEEG_UNIDMFM_BACKEND_ROOT'],
            os.environ['TORCHEEG_UNIDMFM_CHECKPOINT'],
        )
        names = ['Fp1', 'F3', 'C3', 'P3', 'O1']
        native, _ = example.build_unidmfm_backend(root, checkpoint, names)
        model = UniDMFM.from_pretrained(checkpoint, native, names)
        self.assertEqual(len(native.state_dict()), 386)
        self.assertEqual(sum(v.numel() for v in native.state_dict().values()), 29421093)
        self.assertEqual(len(list(native.named_parameters())), 242)
        reference = copy.deepcopy(native)
        for training in [False, True]:
            model.train(training)
            reference.train(training)
            for mode in ['pooled', 'tokens', 'pretraining', 'contrastive']:
                native.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                torch.manual_seed(11)
                left = torch.randn(2, 5, 800, requires_grad=True)
                right = left.detach().clone().requires_grad_()
                torch.manual_seed(27)
                if mode in ['tokens', 'pooled']:
                    actual = {'features': model(left, return_tokens=mode == 'tokens')}
                elif mode == 'pretraining':
                    actual = model.forward_pretraining(left)
                else:
                    actual = {'contrastive_projection': model.forward_contrastive(left)}
                torch.manual_seed(27)
                patches = right.reshape(2, 5, 4, 200)
                if mode in ['tokens', 'pooled']:
                    features = reference.forward_features(patches)
                    expected = {
                        'features': features
                        if mode == 'tokens'
                        else features.mean((1, 2))
                    }
                elif mode == 'pretraining':
                    expected = dict(
                        zip(
                            [
                                'time_prediction',
                                'frequency_prediction',
                                'contrastive_projection',
                            ],
                            reference(patches, mode='full'),
                        )
                    )
                else:
                    expected = {
                        'contrastive_projection': reference(patches, mode='contrastive')
                    }
                for key in actual:
                    torch.testing.assert_close(
                        actual[key], expected[key], rtol=0, atol=0, msg=key
                    )
                sum(v.square().mean() for v in actual.values()).backward()
                sum(v.square().mean() for v in expected.values()).backward()
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(left.grad).all())
                for (name, p), (_, q) in zip(
                    native.named_parameters(), reference.named_parameters()
                ):
                    self.assertEqual(p.grad is None, q.grad is None, name)
                    if p.grad is not None:
                        torch.testing.assert_close(
                            p.grad, q.grad, rtol=0, atol=0, msg=name
                        )
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
        model.eval()
        with torch.no_grad():
            before = model(left)
            changed = left.detach().clone()
            changed[1] += 4
            torch.testing.assert_close(model(changed)[0], before[0], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
