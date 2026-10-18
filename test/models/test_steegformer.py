import copy
from argparse import Namespace
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest

import torch

from torcheeg.models import STEEGFormer


class UnknownMetadata:
    pass


class TestSTEEGFormer(unittest.TestCase):
    def build(self, **kwargs):
        config = dict(hid_channels=16, mlp_channels=32, depth=2, num_heads=4)
        config.update(kwargs)
        return STEEGFormer(**config)

    def test_native_float64_tokens_and_all_parameter_gradients(self):
        path = Path(__file__).parent / 'fixtures' / 'steegformer_reference.json'
        reference = json.loads(path.read_text())

        def tensor(record):
            return torch.tensor(record['values'], dtype=torch.float64).reshape(record['shape'])

        model = STEEGFormer(**reference['config']).double().eval()
        model.load_state_dict({k: tensor(v) for k, v in reference['state_dict'].items()}, strict=True)
        indices = torch.tensor(reference['channel_indices'])
        for case in reference['cases']:
            with self.subTest(normalize=case['normalize']):
                model.zero_grad(set_to_none=True)
                x = tensor(reference['input']).requires_grad_()
                actual = model.forward_all(x, indices, normalize=case['normalize'])
                torch.testing.assert_close(actual, tensor(case['output']), rtol=1e-10, atol=1e-11)
                (actual * tensor(reference['coefficient'])).sum().backward()
                torch.testing.assert_close(x.grad, tensor(case['input_gradient']), rtol=1e-9, atol=1e-10)
                for name, parameter in model.named_parameters():
                    expected = case['parameter_gradients'][name]
                    if expected is None:
                        self.assertIsNone(parameter.grad, name)
                    else:
                        torch.testing.assert_close(parameter.grad, tensor(expected), rtol=1e-9, atol=1e-10, msg=name)

    def test_feature_layout_normalization_and_batch_indices(self):
        model = self.build().eval()
        x = torch.randn(2, 3, 64)
        indices = model.channel_indices(['fp1', ' FP2 ', 'Cz'])
        tokens = model.forward_all(x, indices)
        features = model.forward_features(x, indices)
        self.assertEqual(features.shape, (2, 4, 3, 16))
        torch.testing.assert_close(features.flatten(1, 2), tokens[:, 1:], rtol=0, atol=0)
        torch.testing.assert_close(model(x, indices), tokens[:, 1:].mean(1), rtol=0, atol=0)
        torch.testing.assert_close(model(x, indices), model(x, indices.repeat(2, 1)), rtol=0, atol=0)
        torch.testing.assert_close(model(x, indices), model(x, indices.unsqueeze(0)), rtol=0, atol=0)
        raw = model.forward_all(x, indices, normalize=False)
        torch.testing.assert_close(model.norm(raw), tokens, rtol=0, atol=0)
        self.assertFalse(torch.equal(raw[:, 1:].mean(1), model(x, indices)))

    def test_channel_mapping_never_silently_relabels_sensors(self):
        torch.testing.assert_close(STEEGFormer.channel_indices(['C1', 'Pz', 'Fp1', 'Cz', 'PPO2h']),
                                   torch.tensor([0, 1, 6, 110, 141]))
        for names in ([], 'Cz', ['unknown'], ['Fp1-F7'], ['Fp1-REF'], [None]):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, 'pretrained channel'):
                STEEGFormer.channel_indices(names)
        # The three unnamed slots remain available through explicit integer indices.
        self.assertTrue(torch.isfinite(self.build()(torch.randn(1, 3, 32), torch.tensor([142, 143, 144]))).all())

    def test_invalid_inputs_and_indices_fail_explicitly(self):
        model = self.build(max_patches=4)
        indices = torch.tensor([6, 60])
        for x in (torch.randn(2, 2, 15), torch.randn(2, 2, 17), torch.randn(2, 2, 80),
                  torch.randn(0, 2, 32), torch.randn(2, 0, 32), torch.randn(2, 32),
                  torch.ones(2, 2, 32, dtype=torch.int64), torch.ones(2, 2, 32, dtype=torch.complex64), None):
            with self.subTest(shape=None if x is None else x.shape), self.assertRaisesRegex(ValueError, 'complete patches'):
                model(x, indices)
        x = torch.randn(2, 2, 32)
        for invalid in (None, [6, 60], torch.tensor([6.0, 60.0]), torch.tensor([True, False]),
                        torch.tensor([6]), torch.ones(2, 2, 1, dtype=torch.long), torch.ones(3, 2, dtype=torch.long)):
            with self.subTest(indices=invalid), self.assertRaisesRegex(ValueError, 'index|indices'):
                model(x, invalid)
        for invalid in (torch.tensor([-1, 6]), torch.tensor([6, 145])):
            with self.assertRaises((IndexError, RuntimeError)):
                model(x, invalid)
        with self.assertRaisesRegex(ValueError, 'normalize'):
            model.forward_all(x, indices, normalize=1)

    def test_constructor_validation(self):
        for options in ({'patch_size': 0}, {'depth': True}, {'hid_channels': 15}, {'hid_channels': 18},
                        {'num_heads': 0}, {'mlp_channels': -1}, {'max_patches': 0}, {'num_channel_embeddings': 0},
                        {'qkv_bias': 1}, {'dropout': 1.}, {'attention_dropout': float('nan')},
                        {'drop_path_rate': -1.}, {'norm_eps': float('inf')}, {'norm_eps': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.build(**options)

    def test_zero_inputs_copy_and_complete_state(self):
        model = self.build().double().eval()
        x = torch.zeros(2, 2, 32, dtype=torch.float64, requires_grad=True)
        indices = torch.tensor([[6, 60], [110, 71]])
        output = model.forward_all(x, indices)
        (output * torch.linspace(-1, 1, output.numel(), dtype=output.dtype).reshape_as(output)).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        cloned = copy.deepcopy(model)
        restored = self.build().double().eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(cloned(x, indices), restored(x, indices), rtol=0, atol=0)
        with torch.no_grad():
            cloned.patch_embed.proj.weight.add_(1.)
        self.assertFalse(torch.equal(model.patch_embed.proj.weight, cloned.patch_embed.proj.weight))

    def test_dropout_stochastic_depth_and_native_variant_shapes(self):
        model = self.build(dropout=.5, attention_dropout=.5, drop_path_rate=.5)
        x, indices = torch.randn(4, 2, 32), torch.tensor([6, 60])
        model.eval()
        torch.testing.assert_close(model(x, indices), model(x, indices), rtol=0, atol=0)
        model.train()
        self.assertFalse(torch.equal(model(x, indices), model(x, indices)))
        for width, hidden, depth, heads in ((512, 2048, 8, 8), (768, 3072, 12, 12), (1024, 4096, 24, 16)):
            with self.subTest(width=width), torch.device('meta'):
                variant = STEEGFormer(hid_channels=width, mlp_channels=hidden, depth=depth, num_heads=heads)
                self.assertEqual(variant.enc_channel_emd.channel_transformation.weight.shape, (145, width))
                self.assertEqual(len(variant.state_dict()), 12 * depth + 7)
                self.assertEqual(variant.blocks[-1].attn.qkv.weight.shape, (3 * width, width))

    def test_safe_loading_wrappers_prefix_and_explicit_decoder_exclusion(self):
        model = self.build().eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            for container in (None, 'state_dict', 'model'):
                state = model.state_dict()
                torch.save(state if container is None else {container: state}, path)
                restored = STEEGFormer.from_pretrained(path, hid_channels=16, mlp_channels=32, depth=2, num_heads=4)
                for name, tensor in state.items():
                    torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
            wrapped = {'backbone.' + name: value for name, value in model.state_dict().items()}
            wrapped['backbone.decoder_pred.weight'] = torch.ones(16, 8)
            wrapped['backbone.mask_token'] = torch.ones(1, 1, 8)
            wrapped['other_component.weight'] = torch.ones(1)
            torch.save({'state_dict': wrapped}, path)
            options = dict(prefix='backbone.', hid_channels=16, mlp_channels=32, depth=2, num_heads=4)
            with self.assertRaisesRegex(ValueError, 'extra='):
                STEEGFormer.from_pretrained(path, **options)
            restored = STEEGFormer.from_pretrained(path, exclude_decoder=True, **options)
            x, indices = torch.randn(2, 2, 32), torch.tensor([6, 60])
            torch.testing.assert_close(restored.eval()(x, indices), model(x, indices), rtol=0, atol=0)

    def test_safe_loading_rejects_incomplete_invalid_and_ambiguous_states(self):
        model = self.build()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            states = []
            missing = dict(model.state_dict());missing.pop('norm.weight');states.append(missing)
            extra = dict(model.state_dict());extra['unknown'] = torch.ones(1);states.append(extra)
            mismatch = dict(model.state_dict());mismatch['norm.weight'] = torch.ones(17);states.append(mismatch)
            invalid = dict(model.state_dict());invalid['norm.weight'] = torch.full((16,), float('nan'));states.append(invalid)
            invalid = dict(model.state_dict());invalid['norm.weight'] = torch.ones(16, dtype=torch.int64);states.append(invalid)
            states.extend([{'model': {}, 'state_dict': {}}, {'state_dict': {1: torch.ones(1)}}, [torch.ones(1)]])
            for state in states:
                with self.subTest(state_type=type(state)):
                    torch.save(state, path)
                    with self.assertRaises(ValueError):
                        STEEGFormer.from_pretrained(path, hid_channels=16, mlp_channels=32, depth=2, num_heads=4)

    def test_namespace_opt_in_does_not_allow_other_metadata_or_leak_globals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            state = self.build().state_dict()
            before = torch.serialization.get_safe_globals()
            torch.save({'model': state, 'args': Namespace(epochs=1)}, path)
            options = dict(hid_channels=16, mlp_channels=32, depth=2, num_heads=4)
            with self.assertRaises(pickle.UnpicklingError):
                STEEGFormer.from_pretrained(path, **options)
            if not hasattr(torch.serialization, 'safe_globals'):
                with self.assertRaisesRegex(RuntimeError, 'tensor-only encoder'):
                    STEEGFormer.from_pretrained(path, allow_namespace=True, **options)
                self.assertCountEqual(before, torch.serialization.get_safe_globals())
                return
            STEEGFormer.from_pretrained(path, allow_namespace=True, **options)
            self.assertCountEqual(before, torch.serialization.get_safe_globals())
            torch.save({'model': state, 'args': UnknownMetadata()}, path)
            with self.assertRaises(pickle.UnpicklingError):
                STEEGFormer.from_pretrained(path, allow_namespace=True, **options)
            self.assertCountEqual(before, torch.serialization.get_safe_globals())

    @unittest.skipUnless(os.environ.get('TORCHEEG_TEST_COMPILE') == '1' or os.environ.get('GITHUB_ACTIONS') == 'true',
                         'Enable TORCHEEG_TEST_COMPILE=1 with a C++ compiler; GitHub Actions runs this regression.')
    def test_compiled_training_and_invalid_channel_indices(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(2)
        model = self.build(depth=1).eval()
        reference = copy.deepcopy(model)
        x = torch.randn(2, 2, 32, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        indices = torch.tensor([[6, 60], [110, 71]])
        expected = reference(x, indices)
        coefficient = torch.linspace(-.7, .9, expected.numel()).reshape_as(expected)
        (expected * coefficient).sum().backward()
        compiled = torch.compile(model, backend='inductor', fullgraph=True)
        actual = compiled(y, indices)
        (actual * coefficient).sum().backward()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(y.grad, x.grad, rtol=1e-3, atol=1e-5)
        for (name, left), (_, right) in zip(model.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-3, atol=1e-5, msg=name)
        for value in (-1, 145):
            invalid = indices.clone()
            invalid[0, 0] = value
            with self.subTest(index=value), self.assertRaises((RuntimeError, IndexError)):
                compiled(x.detach(), invalid)


if __name__ == '__main__':
    unittest.main()
