"""Compare against the complete, separately installed pinned Braindecode model."""

import hashlib
import os
from pathlib import Path
import unittest

import torch

from torcheeg.models import EEGInceptionERP


@unittest.skipUnless(
    os.environ.get('TORCHEEG_EEGINCEPTION_REFERENCE'),
    'Dedicated CI supplies the complete pinned Braindecode source.',
)
class TestEEGInceptionERPNative(unittest.TestCase):
    def test_complete_reference_initialization_outputs_gradients_and_updates(self):
        import braindecode
        from braindecode.models import EEGInceptionERP as Native

        root = Path(os.environ['TORCHEEG_EEGINCEPTION_REFERENCE']).resolve()
        self.assertEqual(
            Path(braindecode.__file__).resolve().parent, root / 'braindecode'
        )
        hashes = {
            'models/eeginception_erp.py': '1bc8ce562b1fe3f2cb4cff2a34b48f9441d70621572462c9eaae90356044e139',
            'modules/convolution.py': '9716759ccffcf537f9f1e11912ce0985e0f1247ccb7145b513b61fda44262a87',
            'modules/blocks.py': '36ffa1bb59f1d33641b3469831eec50c2cd1907d6b7f7e021d7a684d03fa9421',
            'modules/layers.py': '04d6ba1ad3837cae04286a9a1b2328459920e3a8c4f612fac9f8ef631a3cb348',
            'functional/initialization.py': '367efbde110c59560361eba716fef7b5dc5db5c6305c50bb3381f3adf5ac6874',
        }
        for name, digest in hashes.items():
            self.assertEqual(
                hashlib.sha256((root / 'braindecode' / name).read_bytes()).hexdigest(),
                digest,
            )
        for electrodes, length, filters, rate, scales, pools in [
            (8, 128, 8, 128, (0.5, 0.25, 0.125), (4, 2, 2, 2)),
            (3, 137, 5, 32, (7 / 32, 5 / 32, 9 / 32), (3, 2, 2, 2)),
        ]:
            torch.manual_seed(19)
            native = Native(
                n_chans=electrodes,
                n_times=length,
                n_outputs=2,
                n_filters=filters,
                sfreq=rate,
                scales_samples_s=scales,
                pooling_sizes=pools,
            )
            native_rng = torch.get_rng_state()
            torch.manual_seed(19)
            model = EEGInceptionERP(
                num_electrodes=electrodes,
                chunk_size=length,
                num_filters=filters,
                sampling_rate=rate,
                scales=scales,
                pool_sizes=pools,
            )
            self.assertTrue(torch.equal(torch.get_rng_state(), native_rng))
            self.assertEqual(set(model.state_dict()), set(native.state_dict()))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(
                    value, native.state_dict()[name], rtol=0, atol=0
                )
            model.load_state_dict(native.state_dict(), strict=True)
            for training in [False, True]:
                model.train(training)
                native.train(training)
                left = torch.randn(3, electrodes, length, requires_grad=True)
                right = left.detach().clone().requires_grad_()
                model.zero_grad(set_to_none=True)
                native.zero_grad(set_to_none=True)
                torch.manual_seed(37)
                actual = model(left)
                actual_rng = torch.get_rng_state()
                torch.manual_seed(37)
                expected = native(right)
                self.assertTrue(torch.equal(torch.get_rng_state(), actual_rng))
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                targets = torch.tensor([0, 1, 0])
                loss = torch.nn.functional.cross_entropy(actual, targets)
                reference = torch.nn.functional.cross_entropy(expected, targets)
                torch.testing.assert_close(loss, reference, rtol=1e-5, atol=1e-6)
                loss.backward()
                reference.backward()
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)
                for name, value in model.named_parameters():
                    self.assertIsNotNone(value.grad, name)
                    torch.testing.assert_close(
                        value.grad,
                        dict(native.named_parameters())[name].grad,
                        rtol=1e-5,
                        atol=1e-6,
                        msg=name,
                    )
                first = torch.optim.Adam(model.parameters(), lr=0.001)
                second = torch.optim.Adam(native.parameters(), lr=0.001)
                first.step()
                second.step()
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(
                        value, native.state_dict()[name], rtol=1e-5, atol=1e-6, msg=name
                    )


if __name__ == '__main__':
    unittest.main()
