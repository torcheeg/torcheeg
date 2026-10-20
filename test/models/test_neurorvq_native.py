"""Optional complete author-model oracle; no author source or weights bundled."""
import copy
import importlib.util
import os
from pathlib import Path
import unittest

import torch

from torcheeg.models import NeuroRVQ


@unittest.skipUnless(os.environ.get('TORCHEEG_NEURORVQ_BACKEND_ROOT') and os.environ.get('TORCHEEG_NEURORVQ_CHECKPOINT'),
                     'Explicit separately licensed native source/checkpoint required.')
class TestNeuroRVQNative(unittest.TestCase):
    def test_complete_features_masked_heads_and_all_gradients(self):
        path = Path(__file__).resolve().parents[2]/'examples'/'examples_eegmat_neurorvq.py'
        spec = importlib.util.spec_from_file_location('neurorvq_example_native', path)
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        native, config = example.build_neurorvq_backend(os.environ['TORCHEEG_NEURORVQ_BACKEND_ROOT'], ['Fp1', 'Fp2'])
        model = NeuroRVQ.from_pretrained(os.environ['TORCHEEG_NEURORVQ_CHECKPOINT'], native,
                                        ['Fp1', 'Fp2'], config['channel_vocab'])
        self.assertEqual(len(native.state_dict()), 330)
        self.assertEqual(sum(v.numel() for v in native.state_dict().values()), 58560632)
        reference = copy.deepcopy(native)
        torch.manual_seed(17)
        values = torch.randn(2, 2, 400)
        temporal = torch.tensor([[254, 255, 254, 255]]).expand(2, -1)
        vocab = config['channel_vocab']
        spatial = torch.tensor([[vocab.index('fp1')]*2+[vocab.index('fp2')]*2]).expand(2, -1)
        for training in [False, True]:
            model.train(training); reference.train(training)
            for masked in [False, True]:
                native.zero_grad(); reference.zero_grad()
                left = values.clone().requires_grad_(); right = values.clone().requires_grad_()
                mask = torch.tensor([[[True, False], [True, False]], [[False, True], [False, True]]])
                if masked:
                    actual = model.forward_masked(left, mask)
                    expected = reference(right.reshape(2, 2, 2, 200), temporal, spatial,
                                         bool_masked_pos=mask.flatten(1), use_for_pretraining=True)
                    self.assertEqual(actual[0].shape, (32, 4, 8192))
                else:
                    actual = (model(left, return_tokens=True),)
                    branches = reference(right.reshape(2, 2, 2, 200), temporal, spatial, return_patch_tokens=True)
                    expected = (torch.cat(branches[:4], -1).reshape(2, 2, 2, 800),)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                sum(a.square().mean() for a in actual).backward()
                sum(b.square().mean() for b in expected).backward()
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
                for (name, a), (_, b) in zip(native.named_parameters(), reference.named_parameters()):
                    self.assertEqual(a.grad is None, b.grad is None, name)
                    if a.grad is not None:
                        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)
                if masked:
                    self.assertTrue(all(getattr(native, 'head_pre_'+str(i)).weight.grad is not None for i in range(1, 33)))
        model.eval()
        with torch.no_grad():
            first = model(values)
            changed = values.clone(); changed[1] += 3
            torch.testing.assert_close(model(changed)[0], first[0], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(left.grad).all())


if __name__ == '__main__':
    unittest.main()
