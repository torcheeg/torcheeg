"""Compare against the complete, separately installed pinned Braindecode model."""

import hashlib
import os
from pathlib import Path
import unittest

import torch

from torcheeg.models import EEGNeX


@unittest.skipUnless(
    os.environ.get('TORCHEEG_EEGNEX_REFERENCE'),
    'Dedicated CI supplies the complete pinned Braindecode source.',
)
class TestEEGNeXNative(unittest.TestCase):
    def test_complete_reference_initialization_outputs_gradients_and_updates(self):
        import braindecode
        from braindecode.models import EEGNeX as Native

        root = Path(os.environ['TORCHEEG_EEGNEX_REFERENCE']).resolve()
        self.assertEqual(
            Path(braindecode.__file__).resolve().parent, root / 'braindecode'
        )
        hashes = {
            'models/eegnex.py': '00b5b9aa80a425e657d50cbca6f285e8b2ac2e259adfc7196127c6f60ac18997',
            'modules/convolution.py': '9716759ccffcf537f9f1e11912ce0985e0f1247ccb7145b513b61fda44262a87',
            'modules/linear.py': 'bcc6e42799b65cbf587b4869bea6b6255389fde8507e4c10afd5814790bdd5ed',
            'modules/parametrization.py': 'b13bd5db5fac769aa9898d8fcf00de6e2ed2784be4170f320e003a3aa243a37e',
        }
        for name, digest in hashes.items():
            self.assertEqual(
                hashlib.sha256((root / 'braindecode' / name).read_bytes()).hexdigest(),
                digest,
            )
        for electrodes, length, filters, depth, kernels, dilations, pools in [
            (22, 1000, (8, 32), 2, (64, 16, 16), (2, 4), (4, 8)),
            (3, 137, (3, 5), 3, (7, 6, 4), (3, 2), (3, 5)),
        ]:
            torch.manual_seed(19)
            native = Native(
                n_chans=electrodes,
                n_times=length,
                n_outputs=2,
                filter_1=filters[0],
                filter_2=filters[1],
                depth_multiplier=depth,
                kernel_block_1_2=kernels[0],
                kernel_block_4=kernels[1],
                kernel_block_5=kernels[2],
                dilation_block_4=dilations[0],
                dilation_block_5=dilations[1],
                avg_pool_block4=pools[0],
                avg_pool_block5=pools[1],
            )
            native_rng = torch.get_rng_state()
            torch.manual_seed(19)
            model = EEGNeX(
                num_electrodes=electrodes,
                chunk_size=length,
                num_classes=2,
                num_filters=filters,
                depth_multiplier=depth,
                temporal_kernels=kernels,
                dilations=dilations,
                pool_sizes=pools,
            )
            self.assertTrue(torch.equal(torch.get_rng_state(), native_rng))
            self.assertEqual(set(model.state_dict()), set(native.state_dict()))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(
                    value, native.state_dict()[name], rtol=0, atol=0
                )
            # Exercise the pullback through active max-norm constraints.
            with torch.no_grad():
                native.block_3[0].parametrizations.weight.original.mul_(20)
                native.final_layer.parametrizations.weight.original.mul_(20)
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
