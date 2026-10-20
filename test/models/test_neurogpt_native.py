"""Optional full original author model, official weights and all gradients."""

import copy
import importlib.util
import os
from pathlib import Path
import unittest

import torch

from torcheeg.models import NeuroGPT


@unittest.skipUnless(
    os.environ.get('TORCHEEG_NEUROGPT_BACKEND_ROOT')
    and os.environ.get('TORCHEEG_NEUROGPT_CHECKPOINT'),
    'Explicit trusted author source and complete official checkpoint required.',
)
class TestNeuroGPTNative(unittest.TestCase):
    def test_complete_original_encoder_gpt_reconstruction_and_every_gradient(self):
        script = (
            Path(__file__).resolve().parents[2]
            / 'examples/examples_bciciv2a_neurogpt.py'
        )
        spec = importlib.util.spec_from_file_location(
            'neurogpt_explicit_example', script
        )
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        native, _ = example.build_neurogpt_backend(
            os.environ['TORCHEEG_NEUROGPT_BACKEND_ROOT'],
            os.environ['TORCHEEG_NEUROGPT_CHECKPOINT'],
            example.CHANNELS,
        )
        model = NeuroGPT.from_pretrained(
            os.environ['TORCHEEG_NEUROGPT_CHECKPOINT'],
            native,
            example.CHANNELS,
            allow_montage_transfer=True,
        )
        self.assertEqual(len(native.state_dict()), 193)
        self.assertEqual(sum(v.numel() for v in native.state_dict().values()), 79528857)
        self.assertEqual(len(model.state_dict()), 194)
        self.assertEqual(
            native.decoder.transformer.config._attn_implementation, 'eager'
        )
        self.assertEqual(
            getattr(native.decoder.transformer, '_attn_implementation', 'eager'),
            'eager',
        )
        reference = copy.deepcopy(native)
        for training in [False, True]:
            model.train(training)
            reference.train(training)
            for mode in ['encoder', 'features', 'reconstruction']:
                native.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                torch.manual_seed(11)
                left = torch.randn(2, 3, 22, 500, requires_grad=True)
                right = left.detach().clone().requires_grad_()
                torch.manual_seed(27)
                if mode == 'encoder':
                    actual = {'features': model.forward_encoder(left)}
                elif mode == 'features':
                    actual = {'features': model(left, return_tokens=True)}
                else:
                    actual = model.forward_reconstruction(left)
                torch.manual_seed(27)
                if mode == 'encoder':
                    expected = {
                        'features': reference.encoder(right).reshape(2, 3, 27, 40)
                    }
                elif mode == 'features':
                    encoded = reference.encoder(right).reshape(2, 3, 1080)
                    embedded = reference.embedder({'inputs': encoded})
                    expected = {
                        'features': reference.decoder(
                            {
                                'inputs_embeds': embedded,
                                'attention_mask': torch.ones(2, 3, dtype=torch.long),
                            }
                        )['outputs']
                    }
                else:
                    losses, prediction = reference.compute_loss(
                        {
                            'inputs': right,
                            'attention_mask': torch.ones(2, 3, dtype=torch.long),
                        },
                        return_outputs=True,
                    )
                    expected = dict(
                        prediction=prediction['outputs'],
                        reconstruction_loss=losses['masking_loss'],
                        loss=losses['loss'],
                    )
                for key in actual:
                    torch.testing.assert_close(
                        actual[key], expected[key], rtol=0, atol=0, msg=key
                    )
                (
                    actual['loss']
                    if mode == 'reconstruction'
                    else actual['features'].square().mean()
                ).backward()
                (
                    expected['loss']
                    if mode == 'reconstruction'
                    else expected['features'].square().mean()
                ).backward()
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
                for key, value in native.state_dict().items():
                    torch.testing.assert_close(
                        value, reference.state_dict()[key], rtol=0, atol=0, msg=key
                    )
        model.eval()
        with torch.no_grad():
            before = model(left, return_tokens=True)
            changed = left.detach().clone()
            changed[1] += 3
            torch.testing.assert_close(
                model(changed, return_tokens=True)[0], before[0], rtol=0, atol=0
            )
            changed = left.detach().clone()
            changed[:, -1] += 3
            torch.testing.assert_close(
                model(changed, return_tokens=True)[:, :-1],
                before[:, :-1],
                rtol=0,
                atol=0,
            )
            self.assertEqual(model(left[:1, :1]).shape, (1, 1024))


if __name__ == '__main__':
    unittest.main()
