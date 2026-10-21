"""Optional complete official weights: native outputs and every gradient."""

import copy
import importlib.util
import os
from pathlib import Path
import unittest

import torch

from torcheeg.models import CSBrain


@unittest.skipUnless(
    os.environ.get('TORCHEEG_CSBRAIN_BACKEND_ROOT')
    and os.environ.get('TORCHEEG_CSBRAIN_CHECKPOINT'),
    'Explicit trusted author source and complete official checkpoint required.',
)
class TestCSBrainNative(unittest.TestCase):
    def test_original_forward_and_all_gradients(self):
        script = (
            Path(__file__).resolve().parents[2] / 'examples/examples_eegmat_csbrain.py'
        )
        spec = importlib.util.spec_from_file_location('csbrain_native_example', script)
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        env = {
            key: os.environ.get(key)
            for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS']
        }
        native, _ = example.build_csbrain_backend(
            os.environ['TORCHEEG_CSBRAIN_BACKEND_ROOT'],
            os.environ['TORCHEEG_CSBRAIN_CHECKPOINT'],
            example.CHANNELS,
        )
        assert env == {key: os.environ.get(key) for key in env}
        weights = torch.load(
            os.environ['TORCHEEG_CSBRAIN_CHECKPOINT'],
            weights_only=True,
            map_location='cpu',
            mmap=True,
        )
        native.load_state_dict({k[7:]: v for k, v in weights.items()}, strict=True)
        reference = copy.deepcopy(native)
        model = CSBrain.from_pretrained(
            os.environ['TORCHEEG_CSBRAIN_CHECKPOINT'], native, example.CHANNELS
        )
        assert len(model.state_dict()) == 296
        index = [
            example.CHANNELS.index(v)
            for v in [
                'Fp1',
                'Fp2',
                'F3',
                'F4',
                'C3',
                'C4',
                'P3',
                'P4',
                'O1',
                'O2',
                'F7',
                'F8',
                'T3',
                'T4',
                'T5',
                'T6',
                'Fz',
                'Cz',
                'Pz',
            ]
        ]
        restore = [index.index(i) for i in range(19)]
        order = list(CSBrain.PRETRAINING_INDICES)
        inverse = [order.index(i) for i in range(19)]
        self.assertEqual(len(native.state_dict()), 295)
        self.assertEqual(sum(v.numel() for v in native.state_dict().values()), 8902000)
        for training in [False, True]:
            model.train(training)
            reference.train(training)
            for mode in ['features', 'reconstruction', 'physical_mask', 'native_mask']:
                model.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                torch.manual_seed(11)
                left = torch.randn(2, 19, 800, requires_grad=True)
                right = left.detach().clone().requires_grad_()
                mask = torch.zeros(2, 19, 4, dtype=torch.bool)
                mask[:, 1::3, ::2] = True
                torch.manual_seed(27)
                if mode == 'features':
                    actual = {'features': model(left, return_tokens=True)}
                elif mode == 'reconstruction':
                    actual = {'prediction': model.forward_reconstruction(left)}
                else:
                    actual = (
                        model.forward_masked(left, mask)
                        if mode == 'physical_mask'
                        else model.forward_native_masked(left, mask)
                    )
                patches = right[:, index].reshape(2, 19, 4, 200)
                target_mask = mask[:, index] if mode == 'physical_mask' else mask
                native_mask = target_mask[:, order] if mode == 'physical_mask' else mask
                torch.manual_seed(27)
                capture = []
                hook = reference.proj_out.register_forward_pre_hook(
                    lambda module, args: capture.append(args[0])
                )
                try:
                    pred = reference(
                        patches,
                        native_mask
                        if mode in ['physical_mask', 'native_mask']
                        else None,
                    )
                finally:
                    hook.remove()
                if mode == 'features':
                    expected = {'features': capture[0]}
                elif mode == 'reconstruction':
                    expected = {'prediction': pred[:, restore]}
                else:
                    expected = dict(
                        prediction=pred[:, restore],
                        target_mask=target_mask[:, restore],
                        hidden_mask=native_mask[:, inverse][:, restore],
                        native_mask=native_mask,
                        loss=torch.nn.functional.mse_loss(
                            pred[target_mask], patches[target_mask]
                        ),
                    )
                for key in actual:
                    torch.testing.assert_close(
                        actual[key], expected[key], rtol=0, atol=0, msg=key
                    )
                actual_loss = (
                    actual['loss']
                    if 'loss' in actual
                    else next(iter(actual.values())).square().mean()
                )
                expected_loss = (
                    expected['loss']
                    if 'loss' in expected
                    else next(iter(expected.values())).square().mean()
                )
                actual_loss.backward()
                expected_loss.backward()
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
                assert torch.isfinite(left.grad).all()
                for (name, p), (_, q) in zip(
                    native.named_parameters(), reference.named_parameters()
                ):
                    assert (p.grad is None) == (q.grad is None), name
                    if p.grad is not None:
                        torch.testing.assert_close(
                            p.grad, q.grad, rtol=0, atol=0, msg=name
                        )
                        assert torch.isfinite(p.grad).all(), name
        model.eval()
        with torch.no_grad():
            before = model(left, return_tokens=True)
            changed = left.detach().clone()
            changed[1] += 10
            torch.testing.assert_close(
                model(changed, return_tokens=True)[0], before[0], rtol=0, atol=0
            )
            physical = model.forward_masked(left, mask)
            changed = left.detach().clone().reshape(2, 19, 4, 200)
            changed[mask] += 10
            torch.testing.assert_close(
                model.forward_masked(changed.reshape(2, 19, 800), mask)['prediction'],
                physical['prediction'],
                rtol=0,
                atol=0,
            )
        # The original fine-tuning architecture also accepts its explicit 22-sensor topology.
        source = __import__('models.CSBrain', fromlist=['CSBrain'])
        names = [
            'Fz',
            'FC3',
            'FC1',
            'FCz',
            'FC2',
            'FC4',
            'C5',
            'C3',
            'C1',
            'Cz',
            'C2',
            'C4',
            'C6',
            'CP3',
            'CP1',
            'CPz',
            'CP2',
            'CP4',
            'P1',
            'Pz',
            'P2',
            'POz',
        ]
        alternate = source.CSBrain(
            brain_regions=[0] + [4] * 17 + [1] * 4,
            sorted_indices=[0, 18, 19, 20, 21] + list(range(1, 18)),
        )
        alternate.load_state_dict({k[7:]: v for k, v in weights.items()}, strict=True)
        direct = copy.deepcopy(alternate).eval()
        transfer = CSBrain(
            alternate, names, backend_channel_names=names, allow_montage_transfer=True
        ).eval()
        x = torch.randn(1, 22, 600)
        with torch.no_grad():
            torch.testing.assert_close(
                transfer.forward_reconstruction(x),
                direct(x.reshape(1, 22, 3, 200)),
                rtol=0,
                atol=0,
            )


if __name__ == '__main__':
    unittest.main()
