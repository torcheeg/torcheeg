from argparse import Namespace
import copy
from functools import partial
import io
import os
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torcheeg.models import LaBraM, LaBraMEncoder
from torcheeg.models.transformer.labram import Attention


CONFIG = dict(
    chunk_size=32,
    patch_size=16,
    out_chans=8,
    num_classes=2,
    embed_dim=16,
    depth=2,
    num_heads=4,
    mlp_ratio=2,
    init_values=0.1,
)


def checkpoint(state):
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    return stream


class ExecutableMetadata:
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return eval, (f"__import__('pathlib').Path({self.path!r}).write_text('executed')",)


class TestLaBraMCompatibility(unittest.TestCase):
    def test_native_spatial_table_bounds_do_not_silently_remap_catalog_entries(self):
        model = LaBraM(**CONFIG).eval()
        with self.assertRaisesRegex(ValueError, 'learned spatial table'):
            model(torch.randn(1, 2, 2, 16), electrodes=['FP1-F3', 'F3-C3'])
        with self.assertRaisesRegex(ValueError, 'explicit electrodes'):
            model(torch.randn(1, 136, 2, 16))
        output = model(torch.randn(1, 2, 2, 16), electrodes=['FP1-F7', 'P8-O2'])
        self.assertTrue(torch.isfinite(output).all())
        ablation = LaBraM(**CONFIG, use_abs_pos_emb=False).eval()
        output = ablation(torch.randn(1, 2, 2, 16), electrodes=['FP1-F3', 'F3-C3'])
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(ablation(torch.randn(1, 136, 2, 16)).shape), (1, 2))

    def test_attention_normalization_view_matches_original_four_dimensional_math(self):
        for qkv_bias in (True, False):
            for normalized in (True, False):
                attention = (
                    Attention(
                        dim=16,
                        num_heads=4,
                        qkv_bias=qkv_bias,
                        qk_norm=partial(nn.LayerNorm, eps=1e-6) if normalized else None,
                    )
                    .double()
                    .eval()
                )
                reference = copy.deepcopy(attention)
                x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
                other = x.detach().clone().requires_grad_()
                bias = (
                    None
                    if reference.q_bias is None
                    else torch.cat(
                        (reference.q_bias, torch.zeros_like(reference.v_bias), reference.v_bias)
                    )
                )
                packed = (
                    F.linear(other, reference.qkv.weight, bias)
                    .reshape(2, 5, 3, 4, 4)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = packed.unbind(0)
                if normalized:
                    q, k = reference.q_norm(q), reference.k_norm(k)
                probabilities = ((q * reference.scale) @ k.transpose(-2, -1)).softmax(-1)
                expected = reference.proj((probabilities @ v).transpose(1, 2).reshape(2, 5, 16))
                actual = attention(x)
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
                coefficient = torch.linspace(
                    -0.7, 1.1, actual.numel(), dtype=actual.dtype
                ).reshape_as(actual)
                (actual * coefficient).sum().backward()
                (expected * coefficient).sum().backward()
                torch.testing.assert_close(x.grad, other.grad, rtol=1e-9, atol=1e-10)
                for (n, p), (m, r) in zip(
                    attention.named_parameters(), reference.named_parameters()
                ):
                    self.assertEqual(n, m)
                    torch.testing.assert_close(p.grad, r.grad, rtol=1e-9, atol=1e-10, msg=n)

    @unittest.skipUnless(
        os.environ.get('GITHUB_ACTIONS') == 'true'
        or os.environ.get('TORCHEEG_TEST_COMPILE') == '1',
        'Optional compiler regression',
    )
    def test_legacy_fullgraph_training_matches_eager_with_optional_positions_and_pooling(self):
        for mean_pooling in (True, False):
            for spatial_positions in (True, False):
                reference = LaBraM(
                    **CONFIG, use_mean_pooling=mean_pooling, use_abs_pos_emb=spatial_positions
                ).eval()
                candidate = copy.deepcopy(reference)
                x = torch.randn(2, 2, 2, 16, requires_grad=True)
                other = x.detach().clone().requires_grad_()
                actual = torch.compile(candidate, fullgraph=True, dynamic=False)(
                    x, electrodes=['FP1', 'C3']
                )
                expected = reference(other, electrodes=['FP1', 'C3'])
                torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-6)
                coefficient = torch.linspace(-0.7, 1.1, actual.numel()).reshape_as(actual)
                (actual * coefficient).sum().backward()
                (expected * coefficient).sum().backward()
                torch.testing.assert_close(x.grad, other.grad, rtol=3e-3, atol=1e-6)
                for (n, p), (m, q) in zip(
                    candidate.named_parameters(), reference.named_parameters()
                ):
                    self.assertEqual(n, m)
                    torch.testing.assert_close(p.grad, q.grad, rtol=3e-3, atol=1e-6, msg=n)

    def test_disabled_spatial_positions_match_zero_spatial_reference_and_all_gradients(self):
        for mean_pooling in (True, False):
            reference = LaBraM(**CONFIG, use_mean_pooling=mean_pooling).double().eval()
            with torch.no_grad():
                reference.pos_embed.zero_()
            model = (
                LaBraM(**CONFIG, use_mean_pooling=mean_pooling, use_abs_pos_emb=False)
                .double()
                .eval()
            )
            model.load_state_dict(
                {n: t for n, t in reference.state_dict().items() if n != 'pos_embed'}, strict=True
            )
            self.assertIsNone(model.pos_embed)
            for mode in ('pooled', 'all', 'patch'):
                model.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                x = torch.randn(2, 2, 2, 16, dtype=torch.float64, requires_grad=True)
                other = x.detach().clone().requires_grad_()
                kwargs = dict(
                    electrodes=['FP1', 'C3'],
                    return_all_tokens=mode == 'all',
                    return_patch_tokens=mode == 'patch',
                )
                actual = model.forward_features(x, **kwargs)
                expected = reference.forward_features(other, **kwargs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                coefficient = torch.linspace(
                    -0.7, 1.1, actual.numel(), dtype=actual.dtype
                ).reshape_as(actual)
                (actual * coefficient).sum().backward()
                (expected * coefficient).sum().backward()
                torch.testing.assert_close(x.grad, other.grad, rtol=0, atol=0)
                params = dict(reference.named_parameters())
                for n, p in model.named_parameters():
                    torch.testing.assert_close(p.grad, params[n].grad, rtol=0, atol=0, msg=n)
                self.assertTrue(torch.isfinite(x.grad).all())
            self.assertEqual(
                tuple(
                    model(
                        torch.randn(2, 2, 2, 16, dtype=torch.float64), electrodes=['FP1', 'C3']
                    ).shape
                ),
                (2, 2),
            )

    def test_raw_and_all_supported_containers_restore_complete_classifiers(self):
        source = LaBraM(**CONFIG).double().eval()
        x = torch.randn(2, 2, 2, 16, dtype=torch.float64)
        for container in (None, 'model', 'module', 'state_dict'):
            model = LaBraM(**CONFIG).double().eval()
            state = source.state_dict()
            payload = state if container is None else {container: state, 'epoch': 3}
            with warnings.catch_warnings(record=True) as recorded:
                result = model.load_pretrained(checkpoint(payload), strict=True)
            self.assertFalse(recorded)
            self.assertEqual(result.missing_keys, [])
            self.assertEqual(result.unexpected_keys, [])
            for n, t in source.state_dict().items():
                torch.testing.assert_close(model.state_dict()[n], t, rtol=0, atol=0)
            torch.testing.assert_close(
                model(x, electrodes=['FP1', 'C3']),
                source(x, electrodes=['FP1', 'C3']),
                rtol=0,
                atol=0,
            )

    def test_student_auto_selection_explicit_prefix_and_no_filter_on_raw_keys(self):
        source = LaBraM(**CONFIG)
        for prefix in ('student.', 'module.student.', ''):
            state = {prefix + n: t for n, t in source.state_dict().items()}
            if prefix == 'student.':
                state['teacher.unused'] = torch.ones(1)
            model = LaBraM(**CONFIG)
            kwargs = {} if prefix == 'student.' else dict(prefix=prefix)
            model.load_pretrained(checkpoint({'model': state}), strict=True, **kwargs)
            for n, t in source.state_dict().items():
                torch.testing.assert_close(model.state_dict()[n], t, rtol=0, atol=0)

    def test_partial_transfer_reports_exact_missing_and_unexpected_names(self):
        model = LaBraM(**CONFIG)
        before = {n: t.clone() for n, t in model.state_dict().items()}
        selected = {'cls_token': torch.full_like(model.cls_token, 0.123), 'unused': torch.ones(2)}
        with self.assertWarnsRegex(UserWarning, 'missing=.*unexpected='):
            result = model.load_pretrained(checkpoint(selected))
        self.assertEqual(set(result.missing_keys), set(before) - {'cls_token'})
        self.assertEqual(result.unexpected_keys, ['unused'])
        torch.testing.assert_close(model.cls_token, selected['cls_token'], rtol=0, atol=0)
        for n, t in before.items():
            if n != 'cls_token':
                torch.testing.assert_close(model.state_dict()[n], t, rtol=0, atol=0)

    def test_native_norm_and_classifier_fc_norm_are_not_silently_renamed(self):
        native = LaBraMEncoder(
            patch_size=16, hid_channels=16, conv_channels=8, mlp_channels=32, depth=2, num_heads=4
        )
        model = LaBraM(**CONFIG)
        fc_norm = copy.deepcopy(model.fc_norm.state_dict())
        with self.assertWarns(UserWarning):
            result = model.load_pretrained(checkpoint(native.state_dict()))
        self.assertEqual(
            set(result.missing_keys), {'fc_norm.weight', 'fc_norm.bias', 'head.weight', 'head.bias'}
        )
        self.assertEqual(set(result.unexpected_keys), {'norm.weight', 'norm.bias'})
        for n, t in fc_norm.items():
            torch.testing.assert_close(model.fc_norm.state_dict()[n], t, rtol=0, atol=0)
        for n, t in native.state_dict().items():
            if n in model.state_dict():
                torch.testing.assert_close(model.state_dict()[n], t, rtol=0, atol=0)

    def test_head_exclusion_is_explicit_and_mismatched_shapes_fail_atomically(self):
        source = LaBraM(**dict(CONFIG, num_classes=3))
        model = LaBraM(**CONFIG)
        before = {n: t.clone() for n, t in model.state_dict().items()}
        with self.assertRaisesRegex(ValueError, 'shape/type mismatch'):
            model.load_pretrained(checkpoint(source.state_dict()))
        for n, t in before.items():
            torch.testing.assert_close(model.state_dict()[n], t, rtol=0, atol=0)
        with self.assertWarns(UserWarning):
            result = model.load_pretrained(checkpoint(source.state_dict()), exclude_head=True)
        self.assertEqual(set(result.missing_keys), {'head.weight', 'head.bias'})
        for n, t in model.state_dict().items():
            expected = before[n] if n.startswith('head.') else source.state_dict()[n]
            torch.testing.assert_close(t, expected, rtol=0, atol=0)

    def test_bad_inputs_nonfinite_shapes_and_strict_failures_do_not_mutate_state(self):
        base = LaBraM(**CONFIG)
        bad_shape = dict(base.state_dict(), **{'blocks.1.mlp.fc1.weight': torch.ones(1)})
        bad_nan = dict(base.state_dict(), cls_token=torch.full_like(base.cls_token, float('nan')))
        bad_int = dict(
            base.state_dict(), cls_token=torch.zeros_like(base.cls_token, dtype=torch.int64)
        )
        cases = [
            (x, {})
            for x in (
                {},
                [],
                {'model': None},
                {1: torch.ones(1)},
                {'unknown': torch.ones(1)},
                {'cls_token': 'not a tensor'},
                {'model': base.state_dict(), 'module': base.state_dict()},
                bad_shape,
                bad_nan,
                bad_int,
            )
        ]
        cases += [
            (base.state_dict(), dict(prefix='absent.')),
            ({'cls_token': base.cls_token.detach()}, dict(strict=True)),
            (base.state_dict(), dict(prefix=2)),
            (base.state_dict(), dict(strict='yes')),
            (base.state_dict(), dict(exclude_head='yes')),
            (base.state_dict(), dict(allow_training_metadata='yes')),
        ]
        for payload, kwargs in cases:
            model = LaBraM(**CONFIG)
            before = {n: t.clone() for n, t in model.state_dict().items()}
            with self.assertRaises(ValueError):
                model.load_pretrained(checkpoint(payload), **kwargs)
            for n, t in before.items():
                torch.testing.assert_close(model.state_dict()[n], t, rtol=0, atol=0)

    def test_partial_warning_as_error_is_also_atomic(self):
        model = LaBraM(**CONFIG)
        before = model.cls_token.detach().clone()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with self.assertRaises(UserWarning):
                model.load_pretrained(checkpoint({'cls_token': torch.ones_like(model.cls_token)}))
        torch.testing.assert_close(model.cls_token, before, rtol=0, atol=0)

    def test_known_author_metadata_requires_explicit_scoped_opt_in(self):
        source = LaBraM(**CONFIG)
        payload = dict(model=source.state_dict(), args=Namespace(epoch=3), loss=np.float64(0.2))
        with self.assertRaises(pickle.UnpicklingError):
            LaBraM(**CONFIG).load_pretrained(checkpoint(payload), strict=True)
        if not hasattr(torch.serialization, 'safe_globals'):
            with self.assertRaisesRegex(RuntimeError, 'tensor-only'):
                LaBraM(**CONFIG).load_pretrained(checkpoint(payload), allow_training_metadata=True)
            return
        before = torch.serialization.get_safe_globals()
        model = LaBraM(**CONFIG)
        model.load_pretrained(checkpoint(payload), strict=True, allow_training_metadata=True)
        self.assertCountEqual(torch.serialization.get_safe_globals(), before)
        for n, t in source.state_dict().items():
            torch.testing.assert_close(model.state_dict()[n], t, rtol=0, atol=0)

    def test_executable_pickle_is_rejected_without_side_effects_or_unsafe_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / 'executed'
            payload = dict(
                model=LaBraM(**CONFIG).state_dict(), metadata=ExecutableMetadata(sentinel)
            )
            for opt_in in (False, True):
                if opt_in and not hasattr(torch.serialization, 'safe_globals'):
                    continue
                before = (
                    torch.serialization.get_safe_globals()
                    if hasattr(torch.serialization, 'get_safe_globals')
                    else []
                )
                with self.assertRaises(pickle.UnpicklingError):
                    LaBraM(**CONFIG).load_pretrained(
                        checkpoint(payload), allow_training_metadata=opt_in
                    )
                self.assertFalse(sentinel.exists())
                if hasattr(torch.serialization, 'get_safe_globals'):
                    self.assertCountEqual(torch.serialization.get_safe_globals(), before)

    def test_safe_load_is_explicit_for_all_runtime_defaults(self):
        model = LaBraM(**CONFIG)
        original = torch.load

        def observe(*args, **kwargs):
            self.assertIs(kwargs['weights_only'], True)
            self.assertEqual(kwargs['map_location'], 'cpu')
            return original(*args, **kwargs)

        with patch('torch.load', side_effect=observe):
            model.load_pretrained(checkpoint(model.state_dict()), strict=True)


if __name__ == '__main__':
    unittest.main()
