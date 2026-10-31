"""Compare complete MSCFormer against a separately supplied pinned package."""

import hashlib
import os
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

from torcheeg.models import MSCFormer


@unittest.skipUnless(
    os.environ.get('TORCHEEG_MSCFORMER_REFERENCE'),
    'Supply the pinned complete Braindecode package for the native oracle.',
)
class TestMSCFormerNative(unittest.TestCase):
    def test_complete_initialization_outputs_all_gradients_and_updates(self):
        root = Path(os.environ['TORCHEEG_MSCFORMER_REFERENCE'])
        self.assertTrue((root / 'braindecode/models/mscformer.py').is_file())
        expected_hashes = {
            'braindecode/models/mscformer.py': '072a83db754ab00ed723ee123448508fe87a1fffb7f2dff7e21f488dd34a462a',
            'braindecode/modules/blocks.py': 'ff7a3055472dcd5991e535ba083cc043f16d80b53ec451114d381bf1835a3fb6',
            'braindecode/modules/layers.py': 'f1fedcec3bf275ec75ac3904055f2349718d6ab81d6828ac03accdae9d7eb574',
            'braindecode/modules/attention.py': 'e2e484ce5ff133cfba1389a6147c3251b501649938852573696c46ceb3ad4538',
            'braindecode/functional/attention.py': '8fc4e29fba668b2011a0ce481cddfb3f8dda317be53432d526af19a9454d12cf',
        }
        for name, digest in expected_hashes.items():
            self.assertEqual(
                hashlib.sha256((root / name).read_bytes()).hexdigest(), digest, name
            )
        sys.path.insert(0, str(root))
        from braindecode.models import MSCFormer as Native

        for seed, config, native_config in [
            (19, {}, dict(n_chans=22, n_times=1000, n_outputs=4)),
            (
                23,
                dict(
                    num_electrodes=3,
                    chunk_size=113,
                    num_classes=2,
                    num_filters=4,
                    temporal_kernels=(16, 32, 64),
                    pool_size=13,
                    heads=3,
                    depth=2,
                    mlp_ratio=2,
                    dropout=0.25,
                    attn_dropout=0.2,
                    pos_dropout=0.15,
                    final_dropout=0.1,
                    attention_scale=0.125,
                ),
                dict(
                    n_chans=3,
                    n_times=113,
                    n_outputs=2,
                    n_filters_time=4,
                    kernel_sizes=(16, 32, 64),
                    pooling_size=13,
                    num_heads=3,
                    num_layers=2,
                    forward_expansion=2,
                    cnn_drop_prob=0.25,
                    att_drop_prob=0.2,
                    att_positional_drop_prob=0.15,
                    final_drop_prob=0.1,
                    attention_scale=0.125,
                ),
            ),
        ]:
            with self.subTest(config=config):
                torch.manual_seed(seed)
                native = Native(**native_config)
                rng = torch.get_rng_state().clone()
                torch.manual_seed(seed)
                model = MSCFormer(**config)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(set(model.state_dict()), set(native.state_dict()))
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(
                        value, native.state_dict()[key], rtol=0, atol=0
                    )
                for training in (False, True):
                    model.train(training)
                    native.train(training)
                    model.zero_grad(set_to_none=True)
                    native.zero_grad(set_to_none=True)
                    left = torch.randn(
                        3, model.num_electrodes, model.chunk_size, requires_grad=True
                    )
                    right = left.detach().clone().requires_grad_()
                    torch.manual_seed(43)
                    actual = model(left)
                    rng = torch.get_rng_state().clone()
                    torch.manual_seed(43)
                    expected = native(right)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    target = torch.arange(3) % model.num_classes
                    loss, other = (
                        F.cross_entropy(actual, target),
                        F.cross_entropy(expected, target),
                    )
                    torch.testing.assert_close(loss, other, rtol=1e-5, atol=1e-6)
                    loss.backward()
                    other.backward()
                    torch.testing.assert_close(
                        left.grad, right.grad, rtol=1e-5, atol=1e-6
                    )
                    for key, value in model.named_parameters():
                        reference = dict(native.named_parameters())[key]
                        self.assertIsNotNone(value.grad, key)
                        self.assertTrue(torch.isfinite(value.grad).all(), key)
                        torch.testing.assert_close(
                            value.grad, reference.grad, rtol=1e-5, atol=1e-6, msg=key
                        )
                    for net in (model, native):
                        torch.optim.AdamW(net.parameters(), lr=0.001).step()
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(
                            value,
                            native.state_dict()[key],
                            rtol=1e-5,
                            atol=1e-6,
                            msg=key,
                        )


if __name__ == '__main__':
    unittest.main()
