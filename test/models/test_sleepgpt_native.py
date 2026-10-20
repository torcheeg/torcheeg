"""Optional complete original author network, official weights and all gradients."""

import copy
import importlib.util
import os
from pathlib import Path
import unittest

import torch

from torcheeg.models import SleepGPT


@unittest.skipUnless(
    os.environ.get('TORCHEEG_SLEEPGPT_BACKEND_ROOT')
    and os.environ.get('TORCHEEG_SLEEPGPT_CHECKPOINT'),
    'Explicit trusted author source and complete official checkpoint required.',
)
class TestSleepGPTNative(unittest.TestCase):
    def test_complete_original_features_both_decoders_and_every_gradient(self):
        script = (
            Path(__file__).resolve().parents[2] / 'examples/examples_eegmat_sleepgpt.py'
        )
        spec = importlib.util.spec_from_file_location(
            'sleepgpt_explicit_example', script
        )
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        root = os.environ['TORCHEEG_SLEEPGPT_BACKEND_ROOT']
        checkpoint = os.environ['TORCHEEG_SLEEPGPT_CHECKPOINT']
        native, _ = example.build_sleepgpt_backend(root, checkpoint, ['C3', 'F3', 'Pz'])
        model = SleepGPT.from_pretrained(checkpoint, native, ['C3', 'F3', 'Pz'])
        self.assertEqual(len(native.state_dict()), 257)
        self.assertEqual(
            sum(v.numel() for v in native.state_dict().values()), 137257856
        )
        self.assertEqual(len(model.state_dict()), 258)
        self.assertFalse(native.hparams.config['use_cb'])
        reference = copy.deepcopy(native)
        for training in [False, True]:
            model.train(training)
            reference.train(training)
            for masked in [False, True]:
                torch.manual_seed(13)
                values = torch.randn(2, 3, 3000)
                left = values.clone().requires_grad_()
                right = values.clone().requires_grad_()
                native.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                torch.manual_seed(19)
                if masked:
                    mask = torch.zeros(2, 3, 15, dtype=torch.bool)
                    mask[:, :, ::3] = True
                    actual = model.forward_masked(left, mask)
                else:
                    actual = {'features': model(left)}
                present = torch.zeros(2, 8)
                present[:, [0, 4, 7]] = 1
                mapped = torch.zeros(2, 8, 3000).index_copy(
                    1, torch.tensor([0, 4, 7]), right
                )
                with torch.no_grad():
                    frequency, _ = reference.transformer.get_fft(mapped, present)
                torch.testing.assert_close(
                    SleepGPT.spectrogram(mapped), frequency, rtol=0, atol=0
                )
                batch = dict(
                    epochs=(mapped, frequency),
                    mask=reference.get_attention_mask(present, present),
                )
                if masked:
                    time_mask = torch.zeros(2, 8, 15).index_copy(
                        1, torch.tensor([0, 4, 7]), mask.float()
                    )
                    batch['random_mask'] = [time_mask.flatten(1)]
                torch.manual_seed(19)
                output = reference.infer(
                    batch, time_mask=masked, stage='train' if masked else 'test'
                )
                if masked:
                    expected = dict(
                        time_prediction=output['cls_feats'],
                        frequency_prediction=output['cls_feats_fft'],
                        time_mask=output['time_mask_patch'].reshape(2, 8, 15),
                        frequency_mask=output['fft_mask_patch'].reshape(2, 8, 15),
                    )
                    expected['time_loss'] = reference.forward_masked_loss(
                        expected['time_prediction'], mapped, output['time_mask_patch']
                    )
                    expected['frequency_loss'] = reference.forward_masked_loss_2D(
                        expected['frequency_prediction'],
                        frequency,
                        output['fft_mask_patch'],
                    )
                    expected['loss'] = (
                        expected['time_loss'] + expected['frequency_loss']
                    )
                else:
                    expected = {
                        'features': torch.cat(
                            [output['time_feats'][:, 0], output['fft_feats'][:, 0]], -1
                        )
                    }
                for name, value in expected.items():
                    torch.testing.assert_close(
                        actual[name], value, rtol=0, atol=0, msg=name
                    )
                loss = actual['loss'] if masked else actual['features'].square().mean()
                target = (
                    expected['loss'] if masked else expected['features'].square().mean()
                )
                loss.backward()
                target.backward()
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(left.grad).all())
                for (name, a), (_, b) in zip(
                    native.named_parameters(), reference.named_parameters()
                ):
                    self.assertEqual(a.grad is None, b.grad is None, name)
                    if a.grad is not None:
                        torch.testing.assert_close(
                            a.grad, b.grad, rtol=0, atol=0, msg=name
                        )
                        self.assertTrue(torch.isfinite(a.grad).all(), name)
                if masked:
                    self.assertIsNotNone(native.Masked_docoder.decoder.weight.grad)
                    self.assertIsNotNone(native.Masked_docoder_fft.decoder.weight.grad)
        # Keep batch layout fixed when verifying isolation; CPU dispatch may
        # change rounding for different batch dimensions.
        model.eval()
        with torch.no_grad():
            before = model(values)
            changed = values.clone()
            changed[1] += 3
            torch.testing.assert_close(model(changed)[0], before[0], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
