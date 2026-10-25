"""Full original author model, obtained explicitly and never bundled."""

import hashlib
import importlib.util
import os
from pathlib import Path
import unittest

import torch

from torcheeg.models import MSVTNet, MSVTNetLoss


@unittest.skipUnless(
    os.environ.get('TORCHEEG_MSVTNET_SOURCE'),
    'Supply the pinned original MSVTNet.py; dedicated CI runs this oracle.',
)
class TestMSVTNetNative(unittest.TestCase):
    def test_original_all_outputs_gradients_buffers_and_optimizer_update(self):
        path = Path(os.environ['TORCHEEG_MSVTNET_SOURCE'])
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            '115a7f4ad1bb1bb7862198c13a1aee86463bc9bdd1ad9bb069881fddda8d549e',
        )
        spec = importlib.util.spec_from_file_location('msvtnet_original', path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        for electrodes, length, kernels in [
            (22, 1000, [15, 31, 63, 125]),
            (3, 113, [16, 32, 64, 126]),
        ]:
            native = original.MSVTNet(nCh=electrodes, nTime=length, C1=kernels)
            model = MSVTNet(
                num_electrodes=electrodes,
                chunk_size=length,
                temporal_kernels=kernels,
                return_branch_logits=True,
            )
            model.load_state_dict(native.state_dict(), strict=True)
            for training in [False, True]:
                model.train(training)
                native.train(training)
                model.zero_grad(set_to_none=True)
                native.zero_grad(set_to_none=True)
                left = torch.randn(3, electrodes, length, requires_grad=True)
                right = left.detach().clone().unsqueeze(1).requires_grad_()
                # Both models consume the same dropout stream.
                torch.manual_seed(17)
                actual = model(left)
                actual_rng = torch.get_rng_state()
                torch.manual_seed(17)
                expected = native(right)
                self.assertTrue(torch.equal(torch.get_rng_state(), actual_rng))
                for x, y in zip(
                    (actual[0],) + actual[1], (expected[0],) + tuple(expected[1])
                ):
                    torch.testing.assert_close(
                        x.log_softmax(-1), y, rtol=1e-5, atol=1e-6
                    )
                labels = torch.tensor([0, 1, 3])
                loss = MSVTNetLoss()(actual, labels)
                reference_loss = original.JointCrossEntoryLoss()(expected, labels)
                torch.testing.assert_close(loss, reference_loss, rtol=1e-5, atol=1e-6)
                loss.backward()
                reference_loss.backward()
                torch.testing.assert_close(
                    left.grad, right.grad.squeeze(1), rtol=1e-5, atol=1e-6
                )
                self.assertEqual(
                    set(dict(model.named_parameters())),
                    set(dict(native.named_parameters())),
                )
                for name, value in model.named_parameters():
                    reference = dict(native.named_parameters())[name]
                    self.assertIsNotNone(value.grad, name)
                    torch.testing.assert_close(
                        value.grad, reference.grad, rtol=1e-5, atol=1e-6, msg=name
                    )
                first = torch.optim.AdamW(model.parameters(), lr=0.001)
                second = torch.optim.AdamW(native.parameters(), lr=0.001)
                first.step()
                second.step()
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(
                        value, native.state_dict()[name], rtol=1e-5, atol=1e-6, msg=name
                    )


if __name__ == '__main__':
    unittest.main()
