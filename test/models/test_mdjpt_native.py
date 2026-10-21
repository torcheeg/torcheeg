"""Complete original neural source/weights; CPU allocation port is explicit.

The CPU reference runs the original loop patcher with only its hardcoded CUDA
allocation redirected to CPU. This is not an unmodified GPU execution claim.
"""

import copy
import importlib
import importlib.util
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from torcheeg.models import MDJPT


def load_original_patcher(root):
    spec = importlib.util.spec_from_file_location(
        'mdjpt_original_loop_oracle', root / 'src/model/MLLA_new.py'
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.to_patch


@unittest.skipUnless(
    os.environ.get('TORCHEEG_MDJPT_BACKEND_ROOT')
    and os.environ.get('TORCHEEG_MDJPT_CHECKPOINT'),
    'Explicit trusted original source and complete official checkpoint required.',
)
class TestMDJPTNative(unittest.TestCase):
    def test_complete_neural_source_cpu_allocation_port_all_outputs_and_gradients(self):
        root = Path(os.environ['TORCHEEG_MDJPT_BACKEND_ROOT']).resolve()
        spec = importlib.util.spec_from_file_location(
            'mdjpt_complete_example',
            Path(__file__).resolve().parents[2] / 'examples/examples_eegmat_mdjpt.py',
        )
        ex = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ex)
        ex.build_mdjpt_backend(root, None, ex.CHANNELS)
        patch_source = importlib.import_module('src.model.MLLA_new')
        original_patches = load_original_patcher(root)

        old_to = torch.Tensor.to
        mapped = [
            'FP1',
            'FP2',
            'F3',
            'F4',
            'F7',
            'F8',
            'T7',
            'T8',
            'C3',
            'C4',
            'P7',
            'P8',
            'P3',
            'P4',
            'O1',
            'O2',
            'FZ',
            'CZ',
            'PZ',
        ]

        def allocation_to(tensor, *args, **kwargs):
            if args and args[0] == 'cuda':
                args = ('cpu',) + args[1:]
            return old_to(tensor, *args, **kwargs)

        rows = []
        for mode in ['mean', 'de', 'projector']:
            backend, _ = ex.build_mdjpt_backend(root, None, ex.CHANNELS)
            weights = torch.load(
                os.environ['TORCHEEG_MDJPT_CHECKPOINT'],
                weights_only=True,
                map_location='cpu',
            )['state_dict']
            backend.load_state_dict(weights, strict=True)
            reference = copy.deepcopy(backend)
            reference.save_fea = mode != 'projector'
            reference.cnn_encoder.saveFea = reference.save_fea
            model = MDJPT.from_pretrained(
                os.environ['TORCHEEG_MDJPT_CHECKPOINT'],
                backend,
                ex.CHANNELS,
                allow_interpolation=True,
                channel_aliases=ex.ALIASES,
                feature_mode=mode,
            )
            for training in [False, True]:
                model.train(training)
                reference.train(training)
                for grouping in ['independent', 'paired']:
                    model.zero_grad(set_to_none=True)
                    reference.zero_grad(set_to_none=True)
                    torch.manual_seed(11)
                    left = torch.randn(
                        (3, 19, 625) if grouping == 'independent' else (2, 3, 19, 625),
                        requires_grad=True,
                    )
                    right = left.detach().clone().requires_grad_()
                    actual = (
                        model(left)
                        if grouping == 'independent'
                        else model.forward_native(left)['features']
                    )
                    with patch.object(
                        patch_source, 'to_patch', original_patches
                    ), patch.object(torch.Tensor, 'to', allocation_to):

                        def infer(x):
                            raw = reference(
                                reference.channel_project(x.unsqueeze(1), mapped)
                            )[0]
                            if mode == 'projector':
                                return raw
                            raw = raw.squeeze(2)
                            return (
                                raw.mean(-1)
                                if mode == 'mean'
                                else 0.5
                                * torch.log(
                                    2 * torch.pi * torch.e * raw.var(-1, unbiased=False)
                                    + 1
                                )
                            )

                        if grouping == 'paired':
                            expected = infer(right.flatten(0, 1))
                        else:
                            values = []
                            for part in right.split(2):
                                n = len(part)
                                values.append(
                                    infer(part if n == 2 else part.expand(2, -1, -1))[
                                        :n
                                    ]
                                )
                            expected = torch.cat(values)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    probe = torch.linspace(0.25, 1.25, actual.numel()).reshape(
                        actual.shape
                    )
                    (actual * probe).mean().backward()
                    (expected * probe).mean().backward()
                    torch.testing.assert_close(
                        left.grad, right.grad, rtol=1e-5, atol=1e-8
                    )
                    count = 0
                    for (name, p), (_, q) in zip(
                        backend.named_parameters(), reference.named_parameters()
                    ):
                        assert (p.grad is None) == (q.grad is None), name
                        if p.grad is not None:
                            torch.testing.assert_close(
                                p.grad, q.grad, rtol=0, atol=0, msg=name
                            )
                            assert torch.isfinite(p.grad).all()
                            count += 1
                    row = dict(
                        mode=mode,
                        training=training,
                        grouping=grouping,
                        output_and_parameter_gradients_exact=True,
                        input_gradient_max_error=(left.grad - right.grad)
                        .abs()
                        .max()
                        .item(),
                        used_native_gradients=count,
                    )
                    rows.append(row)
        # The original multi-domain objectives and coefficients, with true explicit pair ranks.
        backend, _ = ex.build_mdjpt_backend(root, None, ex.CHANNELS)
        model = MDJPT.from_pretrained(
            os.environ['TORCHEEG_MDJPT_CHECKPOINT'],
            backend,
            ex.CHANNELS,
            allow_interpolation=True,
            channel_aliases=ex.ALIASES,
            feature_mode='projector',
        ).train()
        reference = copy.deepcopy(backend)
        torch.manual_seed(63)
        left = [torch.randn(2, 3, 19, 625, requires_grad=True) for _ in range(2)]
        right = [x.detach().clone().requires_grad_() for x in left]
        actual = model.forward_pretraining(left)
        with patch.object(patch_source, 'to_patch', original_patches), patch.object(
            torch.Tensor, 'to', allocation_to
        ):
            encoded = [
                reference(
                    reference.channel_project(x.flatten(0, 1).unsqueeze(1), mapped)
                )
                for x in right
            ]
            contrastive = sum(reference.clisa_loss(v[0])[0] for v in encoded)
            alignment = reference.cda_loss([v[1] for v in encoded])
            expected = dict(
                contrastive_loss=contrastive,
                alignment_loss=alignment,
                loss=contrastive + 0.01 * alignment,
            )
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        actual['loss'].backward()
        expected['loss'].backward()
        for x, y in zip(left, right):
            torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=1e-8)
        for (name, p), (_, q) in zip(
            backend.named_parameters(), reference.named_parameters()
        ):
            assert (p.grad is None) == (q.grad is None), name
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)
                assert torch.isfinite(p.grad).all()
        rows.append(
            dict(
                mode='two_domain_native_objectives',
                aggregate_loss=actual['loss'].item(),
                all_outputs_parameter_gradients_exact=True,
            )
        )

        self.assertEqual(len(rows), 13)
        self.assertEqual(len(backend.state_dict()), 82)

    @unittest.skipUnless(
        torch.cuda.is_available(),
        'Actual CUDA is required; no device redirect is used in this oracle.',
    )
    def test_actual_cuda_original_patcher_outputs_and_all_gradients(self):
        root = Path(os.environ['TORCHEEG_MDJPT_BACKEND_ROOT']).resolve()
        script = (
            Path(__file__).resolve().parents[2] / 'examples/examples_eegmat_mdjpt.py'
        )
        spec = importlib.util.spec_from_file_location('mdjpt_gpu_example', script)
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        rows = []
        for mode in ['mean', 'de', 'projector']:
            backend, _ = example.build_mdjpt_backend(root, None, example.CHANNELS)
            original_patches = load_original_patcher(root)
            patch_source = importlib.import_module('src.model.MLLA_new')
            model = MDJPT.from_pretrained(
                os.environ['TORCHEEG_MDJPT_CHECKPOINT'],
                backend,
                example.CHANNELS,
                allow_interpolation=True,
                channel_aliases=example.ALIASES,
                feature_mode=mode,
            ).cuda()
            reference = copy.deepcopy(backend)
            for training in [False, True]:
                model.train(training)
                reference.train(training)
                model.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                torch.manual_seed(13)
                left = torch.randn(2, 3, 19, 625, device='cuda', requires_grad=True)
                right = left.detach().clone().requires_grad_()
                actual = model.forward_native(left)
                with patch.object(patch_source, 'to_patch', original_patches):
                    raw, cov, tokens = reference(
                        reference.channel_project(
                            right.flatten(0, 1).unsqueeze(1),
                            [
                                'FP1',
                                'FP2',
                                'F3',
                                'F4',
                                'F7',
                                'F8',
                                'T7',
                                'T8',
                                'C3',
                                'C4',
                                'P7',
                                'P8',
                                'P3',
                                'P4',
                                'O1',
                                'O2',
                                'FZ',
                                'CZ',
                                'PZ',
                            ],
                        ),
                        returnMLLAout=True,
                    )
                if mode == 'projector':
                    features = raw
                elif mode == 'mean':
                    features = raw.squeeze(2).mean(-1)
                else:
                    features = 0.5 * torch.log(
                        2 * torch.pi * torch.e * raw.squeeze(2).var(-1, unbiased=False)
                        + 1
                    )
                expected = dict(
                    features=features, covariance_features=cov, channel_tokens=tokens
                )
                expected['projector' if mode == 'projector' else 'feature_map'] = raw
                for key in actual:
                    torch.testing.assert_close(
                        actual[key], expected[key], rtol=1e-5, atol=1e-6, msg=key
                    )
                probe = torch.linspace(
                    0.25, 1.25, actual['features'].numel(), device='cuda'
                ).reshape(actual['features'].shape)
                (actual['features'] * probe).mean().backward()
                (expected['features'] * probe).mean().backward()
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)
                for (name, p), (_, q) in zip(
                    backend.named_parameters(), reference.named_parameters()
                ):
                    self.assertEqual(p.grad is None, q.grad is None, name)
                    if p.grad is not None:
                        torch.testing.assert_close(
                            p.grad, q.grad, rtol=1e-5, atol=1e-6, msg=name
                        )
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                rows.append((mode, training))
        self.assertEqual(len(rows), 6)


if __name__ == '__main__':
    unittest.main()
