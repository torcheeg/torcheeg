"""Optional full-model reference tests, run by the explicit external-backend CI.

No native model code, pretrained values or network download is embedded in
these tests. Supply the separately licensed source and checkpoint explicitly.
"""
import hashlib
import importlib.util
import os
from pathlib import Path
import unittest

import torch

from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LIST
from torcheeg.models import LEAD


ROOT = os.environ.get('TORCHEEG_LEAD_BACKEND_ROOT')
CHECKPOINT = os.environ.get('TORCHEEG_LEAD_CHECKPOINT')
CHECKSUM = '3232e93b5a67a7f3141b16b5ae70b09ae94e65163ccff79755119fbfa598c8db'


@unittest.skipUnless(ROOT and CHECKPOINT, 'Supply the explicit local LEAD source and checkpoint; external-backend CI runs this test.')
class TestLEADNativeBackend(unittest.TestCase):
    def test_complete_official_state_gradients_and_geometry_restore(self):
        path = Path(CHECKPOINT)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), CHECKSUM)
        script = Path(__file__).resolve().parents[2]/'examples/examples_eegmat_lead.py'
        spec = importlib.util.spec_from_file_location('lead_native_example', script)
        example = importlib.util.module_from_spec(spec); spec.loader.exec_module(example)
        channels = MENTAL_ARITHMETIC_CHANNEL_LIST
        native, _ = example.build_lead_backend(ROOT, channels)
        native.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'), strict=True)
        backend, _ = example.build_lead_backend(ROOT, channels)
        wrapped = LEAD.from_pretrained(path, backend, channels)
        self.assertEqual(len(native.state_dict()), 367)
        self.assertEqual(len(wrapped.state_dict()), 370)
        metadata = torch.tensor([[0, 0, 200, 0]]).expand(2, -1)
        for training in [False, True]:
            native.train(training); wrapped.train(training)
            native.zero_grad(set_to_none=True); wrapped.zero_grad(set_to_none=True)
            x = torch.sin(torch.arange(2*19*800).float()/17).reshape(2, 19, 800).requires_grad_()
            xx = x.detach().clone().requires_grad_()
            torch.manual_seed(71); expected = native.encode(x.transpose(1, 2), metadata)
            torch.manual_seed(71); actual = wrapped(xx)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            expected.square().mean().backward(); actual.square().mean().backward()
            torch.testing.assert_close(xx.grad, x.grad, rtol=0, atol=0)
            native_parameters = dict(native.named_parameters())
            self.assertEqual(set(native_parameters), set(dict(wrapped.backend.named_parameters())))
            for name, p in wrapped.backend.named_parameters():
                target = native_parameters[name].grad
                if target is None: self.assertIsNone(p.grad, name)
                else: torch.testing.assert_close(p.grad, target, rtol=0, atol=0, msg=name)
        wrapped.eval()
        eeg = torch.cos(torch.arange(19*800).float()/31).reshape(1, 19, 800)
        state = {name: value.clone() for name, value in wrapped.state_dict().items()}
        target = wrapped(eeg)
        other, _ = example.build_lead_backend(ROOT, channels)
        with torch.no_grad():
            getattr(other.enc_embedding, other.enc_embedding.coord_buffer_names[0]).add_(1.)
        restored = LEAD(other, channels).eval().to_empty(device='cpu')
        restored.load_state_dict(state, strict=True)
        torch.testing.assert_close(restored(eeg), target, rtol=0, atol=0)
        torch.testing.assert_close(restored.electrode_coords, wrapped.electrode_coords, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
