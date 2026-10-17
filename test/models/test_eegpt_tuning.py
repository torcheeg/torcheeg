import copy
import itertools
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from torcheeg.models import (EEGPT, EEGPTClassifier, add_eegpt_adapters,
                             add_eegpt_prefix, add_eegpt_lora)


METHODS = [(add_eegpt_adapters, {'adapter_size': 4, 'dropout': 0.2}),
           (add_eegpt_prefix, {'prefix_length': 3}),
           (add_eegpt_lora, {'rank': 2, 'dropout': 0.2})]


def is_tuning(name):
    return 'adapter_' in name or '.prefix_' in name or '.lora_' in name


class TestEEGPTTuning(unittest.TestCase):
    def encoder(self, **kwargs):
        parameters = dict(num_electrodes=3, channel_names=['C3', 'CZ', 'C4'], chunk_size=128,
                          patch_size=32, hid_channels=16, embed_num=2, depth=2, num_heads=4)
        parameters.update(kwargs)
        return EEGPT(**parameters)

    def test_adapter_and_lora_insertion_preserve_eval_outputs_and_base_owners(self):
        for method, kwargs in [METHODS[0], METHODS[2]]:
            with self.subTest(method=method.__name__):
                encoder = self.encoder().eval()
                before = dict(encoder.named_parameters())
                eeg = torch.randn(2, 3, 128)
                expected = encoder(eeg)
                self.assertIs(method(encoder, **kwargs), encoder)
                torch.testing.assert_close(encoder(eeg), expected, rtol=0, atol=0)
                parameters = dict(encoder.named_parameters())
                for name, value in before.items():
                    self.assertIs(parameters[name], value)
                    self.assertFalse(value.requires_grad)
                self.assertTrue(all(parameter.requires_grad for name, parameter in parameters.items() if is_tuning(name)))
                self.assertTrue(all(not module.training for module in encoder.modules()))

    def test_adapter_blocks_have_independent_active_paths(self):
        encoder = self.encoder().double()
        add_eegpt_adapters(encoder, adapter_size=4, dropout=0, freeze_base=False)
        first, second = encoder.blocks
        self.assertIsNot(first.attn, second.attn)
        self.assertIsNot(first.adapter_attn, second.adapter_attn)
        block = first
        x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            for adapter in [block.adapter_attn, block.adapter_mlp]:
                adapter.up_project.weight.copy_(torch.linspace(-.1, .2, 64).reshape(16, 4))
        def adapter(value, module):
            hidden = F.gelu(F.linear(value, module.down_project.weight, module.down_project.bias))
            return value + F.linear(hidden, module.up_project.weight, module.up_project.bias)
        y = x + block.attn(block.norm1(x))
        y = adapter(y, block.adapter_attn)
        expected = adapter(y + block.mlp(block.norm2(y)), block.adapter_mlp)
        torch.testing.assert_close(block(x), expected, rtol=1e-12, atol=1e-12)
        eeg = torch.randn(2, 3, 128, dtype=torch.float64)
        output = encoder(eeg)
        (output * torch.linspace(.1, 1, output.numel()).reshape(output.shape)).square().mean().backward()
        for index, candidate in enumerate(encoder.blocks):
            for name, parameter in candidate.named_parameters():
                if 'adapter_' in name:
                    self.assertIsNotNone(parameter.grad, (index, name))
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_lora_matches_independent_arithmetic_and_all_gradients(self):
        encoder = self.encoder().double()
        add_eegpt_lora(encoder, rank=2, alpha=3, dropout=0, blocks=[0],
                        target_modules=['attn.qkv'], freeze_base=False)
        linear = encoder.blocks[0].attn.qkv
        with torch.no_grad():
            linear.lora_B.copy_(torch.linspace(-.2, .3, linear.lora_B.numel()).reshape(linear.lora_B.shape))
        x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
        other = x.detach().clone().requires_grad_(True)
        state = {name: parameter.detach().clone().requires_grad_(True) for name, parameter in linear.named_parameters()}
        expected = other @ state['weight'].T + state['bias'] + (other @ state['lora_A'] @ state['lora_B']) * 1.5
        actual = linear(x)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(x.grad, other.grad, rtol=1e-10, atol=1e-12)
        for name, parameter in linear.named_parameters():
            torch.testing.assert_close(parameter.grad, state[name].grad, rtol=1e-10, atol=1e-12)

    def test_lora_input_dropout_is_registered_and_eval_is_deterministic(self):
        encoder = self.encoder()
        add_eegpt_lora(encoder, rank=2, dropout=.8, blocks=[0], target_modules=['attn.qkv'])
        linear = encoder.blocks[0].attn.qkv
        with torch.no_grad():
            linear.lora_B.fill_(.2)
        x = torch.randn(4, 5, 16)
        torch.manual_seed(42)
        actual = linear(x)
        torch.manual_seed(42)
        expected = F.linear(x, linear.weight, linear.bias) + (F.dropout(x, p=.8) @ linear.lora_A @ linear.lora_B) * 4
        torch.testing.assert_close(actual, expected)
        self.assertIn(linear.dropout, list(encoder.modules()))
        encoder.eval()
        self.assertFalse(linear.dropout.training)
        torch.testing.assert_close(linear(x), linear(x), rtol=0, atol=0)
        encoder.train()
        self.assertFalse(torch.equal(linear(x), linear(x)))

    def test_prefix_matches_explicit_softmax_and_all_gradients(self):
        encoder = self.encoder().double()
        add_eegpt_prefix(encoder, prefix_length=3, blocks=[0], freeze_base=False)
        attention = encoder.blocks[0].attn
        x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
        other = x.detach().clone().requires_grad_(True)
        state = {name: parameter.detach().clone().requires_grad_(True) for name, parameter in attention.named_parameters()}
        q, k, v = F.linear(other, state['qkv.weight'], state['qkv.bias']).reshape(2, 5, 3, 4, 4).permute(2, 0, 3, 1, 4)
        k = torch.cat([state['prefix_key'].expand(2, -1, -1, -1), k], dim=-2)
        v = torch.cat([state['prefix_value'].expand(2, -1, -1, -1), v], dim=-2)
        weights = (q @ k.transpose(-2, -1) / 2).softmax(-1)
        expected = F.linear((weights @ v).transpose(1, 2).reshape(2, 5, 16), state['proj.weight'], state['proj.bias'])
        actual = attention(x)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(attention(x, return_attention=True), weights, rtol=1e-10, atol=1e-12)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(x.grad, other.grad, rtol=1e-9, atol=1e-12)
        for name, parameter in attention.named_parameters():
            torch.testing.assert_close(parameter.grad, state[name].grad, rtol=1e-9, atol=1e-12)

    def test_attention_inspection_includes_prefix_keys(self):
        encoder = self.encoder(return_attention_layer=1).eval()
        add_eegpt_prefix(encoder, prefix_length=3)
        add_eegpt_adapters(encoder, adapter_size=4)
        output = encoder(torch.randn(2, 3, 128))
        self.assertEqual(output.shape, (8, 4, 5, 8))
        torch.testing.assert_close(output.sum(-1), torch.ones(8, 4, 5))

    def test_selective_blocks_and_lora_exact_targets(self):
        encoder = self.encoder()
        add_eegpt_adapters(encoder, adapter_size=4, blocks=[1])
        self.assertFalse(hasattr(encoder.blocks[0], 'adapter_attn'))
        self.assertTrue(hasattr(encoder.blocks[1], 'adapter_attn'))
        add_eegpt_lora(encoder, rank=2, blocks=[1], target_modules=['attn.proj'])
        self.assertTrue(hasattr(encoder.blocks[1].attn.proj, 'lora_A'))
        self.assertFalse(hasattr(encoder.blocks[1].attn.qkv, 'lora_A'))
        self.assertFalse(hasattr(encoder.blocks[1].adapter_attn.up_project, 'lora_A'))
        self.assertFalse(hasattr(encoder.blocks[0].attn.proj, 'lora_A'))

    def test_all_composition_orders_preserve_trainable_tuning_parameters(self):
        for methods in itertools.permutations(METHODS):
            with self.subTest(methods=[method.__name__ for method, _ in methods]):
                encoder = self.encoder()
                for method, kwargs in methods:
                    method(encoder, **kwargs)
                self.assertTrue(all(parameter.requires_grad == is_tuning(name) for name, parameter in encoder.named_parameters()))
                output = encoder(torch.randn(2, 3, 128))
                self.assertEqual(output.shape, (2, 4, 2, 16))
                output.square().mean().backward()
                self.assertTrue(all(parameter.grad is not None for name, parameter in encoder.named_parameters() if is_tuning(name)))

    def test_optimizer_updates_tuning_and_head_but_preserves_base_weights(self):
        for method, kwargs in METHODS:
            with self.subTest(method=method.__name__):
                torch.manual_seed(42)
                model = EEGPTClassifier(self.encoder(), dropout=0)
                base = {name: value.clone() for name, value in model.encoder.state_dict().items()}
                self.assertIs(method(model, **kwargs), model)
                before = {name: parameter.detach().clone() for name, parameter in model.encoder.named_parameters() if is_tuning(name)}
                optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3)
                for _ in range(2):
                    optimizer.zero_grad()
                    eeg = torch.randn(4, 3, 128, requires_grad=True)
                    loss = F.cross_entropy(model(eeg), torch.tensor([0, 1, 0, 1]))
                    loss.backward()
                    self.assertTrue(torch.isfinite(eeg.grad).all())
                    for name, parameter in model.encoder.named_parameters():
                        if is_tuning(name):
                            self.assertIsNotNone(parameter.grad, name)
                            self.assertTrue(torch.isfinite(parameter.grad).all())
                        else:
                            self.assertIsNone(parameter.grad, name)
                    optimizer.step()
                self.assertTrue(any(not torch.equal(before[name], parameter) for name, parameter in model.encoder.named_parameters() if name in before))
                for name, value in base.items():
                    torch.testing.assert_close(model.encoder.state_dict()[name], value, rtol=0, atol=0)
                self.assertTrue(all(parameter.requires_grad for parameter in model.task_head.parameters()))

    def test_state_roundtrip_and_deepcopy_have_no_external_parameter_aliases(self):
        for method, kwargs in METHODS:
            with self.subTest(method=method.__name__), tempfile.TemporaryDirectory() as directory:
                encoder = self.encoder().eval()
                method(encoder, **kwargs)
                with torch.no_grad():
                    for name, parameter in encoder.named_parameters():
                        if is_tuning(name):
                            parameter.add_(torch.linspace(-.1, .2, parameter.numel()).reshape(parameter.shape))
                eeg = torch.randn(2, 3, 128)
                expected = encoder(eeg)
                path = Path(directory) / 'state.pt'
                torch.save(encoder.state_dict(), path)
                restored = self.encoder().eval()
                method(restored, **kwargs)
                restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
                torch.testing.assert_close(restored(eeg), expected)
                other = copy.deepcopy(encoder)
                for left, right in zip(encoder.parameters(), other.parameters()):
                    self.assertNotEqual(left.data_ptr(), right.data_ptr())
                torch.testing.assert_close(other(eeg), expected)
                other(eeg).square().mean().backward()
                self.assertTrue(all(parameter.grad is not None for parameter in other.parameters() if parameter.requires_grad))
                self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))

    def test_insertion_preserves_dtype_and_individual_base_module_modes(self):
        for method, kwargs in METHODS:
            with self.subTest(method=method.__name__):
                encoder = self.encoder().double().train()
                encoder.blocks[0].mlp.drop.eval()
                encoder.blocks[0].attn.proj_drop.eval()
                method(encoder, **kwargs)
                self.assertFalse(encoder.blocks[0].mlp.drop.training)
                self.assertFalse(encoder.blocks[0].attn.proj_drop.training)
                self.assertTrue(all(parameter.dtype == torch.float64 for parameter in encoder.parameters()))
                self.assertEqual(encoder(torch.randn(2, 3, 128, dtype=torch.float64)).dtype, torch.float64)
                encoder.float()
                self.assertEqual(encoder(torch.randn(2, 3, 128)).dtype, torch.float32)

    def test_freeze_false_preserves_existing_parameter_flags(self):
        for method, kwargs in METHODS:
            with self.subTest(method=method.__name__):
                encoder = self.encoder()
                encoder.summary_token.requires_grad_(False)
                before = dict(encoder.named_parameters())
                expected = {name: value.requires_grad for name, value in before.items()}
                method(encoder, freeze_base=False, **kwargs)
                for name, parameter in before.items():
                    self.assertEqual(parameter.requires_grad, expected[name])

    def test_duplicate_requests_fail_before_partial_modification(self):
        for method, kwargs in METHODS:
            with self.subTest(method=method.__name__):
                encoder = self.encoder()
                method(encoder, blocks=[0], **kwargs)
                keys = set(encoder.state_dict())
                flags = [parameter.requires_grad for parameter in encoder.parameters()]
                with self.assertRaises(ValueError):
                    method(encoder, blocks=[1, 0], **kwargs)
                self.assertEqual(keys, set(encoder.state_dict()))
                self.assertEqual(flags, [parameter.requires_grad for parameter in encoder.parameters()])

    def test_invalid_settings_do_not_modify_encoder(self):
        cases = [(add_eegpt_adapters, dict(adapter_size=0)), (add_eegpt_adapters, dict(dropout=1)),
                 (add_eegpt_prefix, dict(prefix_length=True)), (add_eegpt_lora, dict(rank=0)),
                 (add_eegpt_lora, dict(alpha=float('nan'))), (add_eegpt_lora, dict(alpha=-1)),
                 (add_eegpt_lora, dict(target_modules=['qkv'])), (add_eegpt_lora, dict(target_modules=[])),
                 (add_eegpt_lora, dict(target_modules=['attn.qkv', 'attn.qkv']))]
        cases += [(method, dict(blocks=blocks)) for method, _ in METHODS for blocks in [[], [True], [-1], [2], [0, 0], '0']]
        cases += [(method, dict(freeze_base=1)) for method, _ in METHODS]
        for method, kwargs in cases:
            with self.subTest(method=method.__name__, kwargs=kwargs):
                encoder = self.encoder()
                before = dict(encoder.named_parameters())
                with self.assertRaises(ValueError):
                    method(encoder, **kwargs)
                self.assertEqual(set(before), set(dict(encoder.named_parameters())))
                self.assertTrue(all(parameter.requires_grad for parameter in encoder.parameters()))
        for method, kwargs in METHODS:
            with self.subTest(method=method.__name__), self.assertRaises(ValueError):
                method(nn.Linear(3, 2), **kwargs)
            model = EEGPTClassifier(self.encoder(), freeze_encoder=True)
            with self.assertRaises(ValueError):
                method(model, **kwargs)
            self.assertFalse(any(is_tuning(name) for name, _ in model.named_parameters()))

    def test_export_save_load_for_individual_and_combined_tuning(self):
        for methods in [[item] for item in METHODS] + [METHODS]:
            with self.subTest(methods=[method.__name__ for method, _ in methods]), tempfile.TemporaryDirectory() as directory:
                encoder = self.encoder()
                for method, kwargs in methods:
                    method(encoder, **kwargs)
                model = EEGPTClassifier(encoder).eval().requires_grad_(False)
                eeg = torch.randn(2, 3, 128)
                exported = torch.export.export(model, (eeg,), strict=True)
                path = Path(directory) / 'tuning.pt2'
                torch.export.save(exported, path)
                torch.testing.assert_close(torch.export.load(path).module()(eeg), model(eeg))


if __name__ == '__main__':
    unittest.main()
