import copy
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from torcheeg.models import (CBraMod, CBraModClassifier, CBraModRegression, CBraModEmbedding,
                             add_cbramod_adapters, add_cbramod_prompts, add_cbramod_lora)
from torcheeg.models.transformer import add_cbramod_lora as TransformerImport
from torcheeg.models.transformer._tuning import _LoRALinear
from torcheeg.models.transformer.eegpt_tuning import _LoRALinear as EEGPTLoRALinear


CONFIG = dict(patch_size=25, hid_channels=16, mlp_channels=32, depth=2,
              num_heads=4, dropout=0., position_kernel=(3, 3))
TARGETS = ('linear1', 'linear2') + tuple(axis + '.' + name
    for axis in ('self_attn_s', 'self_attn_t')
    for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'))


def encoder():
    return CBraMod(**CONFIG)


def task(kind=CBraModClassifier, **kwargs):
    return kind(encoder(), 3, 128, num_patches=4, **kwargs)


def is_tuning(name):
    return '.lora_' in name or '.adapter.' in name or name.endswith('prompt_tokens')


class TestCBraModLoRA(unittest.TestCase):
    def test_public_import_zero_insertion_and_original_parameter_owners(self):
        self.assertIs(add_cbramod_lora, TransformerImport)
        self.assertIs(_LoRALinear, EEGPTLoRALinear)
        for targets in [('linear1', 'linear2'), TARGETS] + [(name,) for name in TARGETS]:
            with self.subTest(targets=targets):
                model = encoder().double().eval()
                original = dict(model.named_parameters())
                x = torch.randn(2, 3, 4, 25, dtype=torch.float64)
                expected = model(x)
                self.assertIs(add_cbramod_lora(model, rank=2, target_modules=targets), model)
                torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
                after = dict(model.named_parameters())
                for name, parameter in original.items():
                    self.assertIs(after[name], parameter)
                    self.assertFalse(parameter.requires_grad)
                self.assertTrue(all(p.requires_grad for n, p in after.items() if '.lora_' in n))
                self.assertTrue(all(p.dtype == torch.float64 for p in model.parameters()))
                self.assertTrue(all(not m.training for m in model.modules()))
                self.assertTrue(all(torch.count_nonzero(p) == 0 for n, p in after.items() if n.endswith('lora_B')))

    def test_independent_native_mha_dense_update_output_and_gradient_reference(self):
        for axis in ('self_attn_s', 'self_attn_t'):
            with self.subTest(axis=axis):
                model = encoder().double().eval()
                targets = tuple(axis + '.' + name for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'))
                add_cbramod_lora(model, rank=3, alpha=7.5, dropout=0., blocks=[0], target_modules=targets)
                attention = getattr(model.encoder.layers[0], axis)
                with torch.no_grad():
                    for name, parameter in attention.named_parameters():
                        if name.endswith('lora_B'):
                            parameter.copy_(torch.linspace(-.2, .3, parameter.numel()).reshape(parameter.shape))
                x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
                y = x.detach().clone().requires_grad_()
                actual = attention(x)

                def delta(module):
                    self.assertEqual(module.scaling, 2.5)
                    return (module.lora_A @ module.lora_B).T * 2.5
                weights = torch.cat([weight + delta(getattr(attention, 'lora_' + name))
                    for weight, name in zip(attention.in_proj_weight.chunk(3), ('q', 'k', 'v'))])
                output_weight = attention.out_proj.weight + delta(attention.out_proj)
                sequence = y.transpose(0, 1)
                expected = F.multi_head_attention_forward(sequence, sequence, sequence,
                    embed_dim_to_check=8, num_heads=attention.num_heads,
                    in_proj_weight=weights, in_proj_bias=attention.in_proj_bias,
                    bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0.,
                    out_proj_weight=output_weight, out_proj_bias=attention.out_proj.bias,
                    training=False, need_weights=False)[0].transpose(0, 1)
                torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
                coefficients = torch.linspace(-1, 1, actual.numel(), dtype=torch.float64).reshape(actual.shape)
                parameters = [p for n, p in attention.named_parameters() if '.lora_' in n or n.startswith('lora_')]
                first = torch.autograd.grad((actual * coefficients).sum(), [x] + parameters)
                second = torch.autograd.grad((expected * coefficients).sum(), [y] + parameters)
                self.assertEqual(len(first), 9)
                for left, right in zip(first, second):
                    torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-12)
                    self.assertTrue(torch.isfinite(left).all())

    def test_default_selection_exact_parameter_count_and_partial_blocks(self):
        model = encoder()
        add_cbramod_lora(model, rank=2, blocks=[1])
        self.assertIs(type(model.encoder.layers[0].linear1), nn.Linear)
        self.assertIs(type(model.encoder.layers[1].linear1), _LoRALinear)
        self.assertFalse(any('self_attn' in n and 'lora' in n for n, _ in model.named_parameters()))
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 192)
        model = encoder()
        add_cbramod_lora(model, rank=2, target_modules=TARGETS)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 896)
        self.assertEqual(model.encoder.layers[0].linear1.lora_A.shape, (16, 2))
        self.assertEqual(model.encoder.layers[0].linear1.lora_B.shape, (2, 32))
        self.assertEqual(model.encoder.layers[0].self_attn_s.lora_q.lora_A.shape, (8, 2))

    def test_dropout_receives_input_features_before_low_rank_projection(self):
        model = encoder().double()
        add_cbramod_lora(model, rank=2, alpha=4., dropout=.5, blocks=[0], target_modules=('linear1',))
        linear = model.encoder.layers[0].linear1
        with torch.no_grad():
            linear.lora_B.normal_(std=.1)
        x = torch.randn(2, 3, 16, dtype=torch.float64)
        mask = (torch.arange(16) % 2).double() * 2
        with patch.object(linear.dropout, 'forward', side_effect=lambda values: values * mask) as observe:
            actual = linear(x)
        self.assertIs(observe.call_args.args[0], x)
        expected = F.linear(x, linear.weight, linear.bias) + ((x * mask) @ linear.lora_A @ linear.lora_B) * 2.
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_two_optimizer_steps_update_all_low_rank_paths_and_no_base_state(self):
        model = task(dropout=0.).double()
        original = {name: value.clone() for name, value in model.backbone.state_dict().items()}
        add_cbramod_lora(model, rank=2, alpha=4., dropout=0., target_modules=TARGETS)
        updates = [(name, p) for name, p in model.backbone.named_parameters() if name.endswith('lora_B')]
        self.assertEqual(len(updates), 20)
        head = model.task_head.output.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        x = torch.randn(3, 3, 128, dtype=torch.float64, requires_grad=True)
        for step in range(2):
            optimizer.zero_grad()
            F.cross_entropy(model(x), torch.tensor([0, 1, 0])).backward()
            self.assertGreater(x.grad.norm().item(), 0)
            for name, parameter in updates:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(parameter.grad.norm().item(), 0, name)
            if step:
                self.assertTrue(all(p.grad.norm() > 0 for name, p in model.backbone.named_parameters() if name.endswith('lora_A')))
            optimizer.step()
        self.assertTrue(all(torch.count_nonzero(p) > 0 for _, p in updates))
        self.assertFalse(torch.equal(head, model.task_head.output.weight))
        for name, value in original.items():
            torch.testing.assert_close(model.backbone.state_dict()[name], value, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for name, p in model.backbone.named_parameters() if '.lora_' not in name))

    def test_copy_and_strict_augmented_state_restoration(self):
        model = task(dropout=0.).double().eval()
        add_cbramod_lora(model, rank=2, dropout=0., target_modules=TARGETS)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith('lora_B'):
                    parameter.normal_(std=.02)
        x = torch.randn(2, 3, 128, dtype=torch.float64)
        expected = model(x)
        copied = copy.deepcopy(model)
        torch.testing.assert_close(copied(x), expected, rtol=0, atol=0)
        self.assertNotEqual(copied.backbone.encoder.layers[0].linear1.lora_B.data_ptr(), model.backbone.encoder.layers[0].linear1.lora_B.data_ptr())
        with torch.no_grad():
            copied.backbone.encoder.layers[0].linear1.lora_B.add_(.1)
        self.assertGreater((copied(x) - expected).abs().max().item(), 1e-8)
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'augmented.pt'
            torch.save(model.state_dict(), path)
            restored = task(dropout=0.).double().eval()
            add_cbramod_lora(restored, rank=2, dropout=0., target_modules=TARGETS)
            restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
            torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
            with self.assertRaises(RuntimeError):
                task().load_state_dict(torch.load(path, weights_only=True), strict=True)

    def test_all_adapter_prompt_lora_composition_orders_and_state_keys(self):
        shared_state, expected = None, None
        inputs = torch.randn(2, 3, 128, dtype=torch.float64)
        for order in itertools.permutations(('adapter', 'prompt', 'lora')):
            with self.subTest(order=order):
                model = task(dropout=0.).double().eval()
                model.task_head.output.bias.requires_grad_(False)
                for name in order:
                    if name == 'adapter':
                        add_cbramod_adapters(model, adapter_size=4, dropout=0.)
                    elif name == 'prompt':
                        add_cbramod_prompts(model, num_prompts=2)
                    else:
                        add_cbramod_lora(model, rank=2, dropout=0., target_modules=TARGETS)
                self.assertTrue(all(p.requires_grad == is_tuning(n) for n, p in model.backbone.named_parameters()))
                self.assertFalse(model.task_head.output.bias.requires_grad)
                if shared_state is None:
                    shared_state = {n: v.clone() for n, v in model.state_dict().items()}
                    expected = model(inputs).detach()
                else:
                    self.assertEqual(set(shared_state), set(model.state_dict()))
                    model.load_state_dict(shared_state, strict=True)
                    torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
                model(inputs).square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))

    def test_incremental_attention_targets_preserve_existing_lora_objects(self):
        for order in [('out_proj', 'q_proj', 'k_proj', 'v_proj'), ('q_proj', 'v_proj', 'out_proj', 'k_proj')]:
            model = encoder().eval()
            original = dict(model.named_parameters())
            first = None
            for name in order:
                add_cbramod_lora(model, rank=2, target_modules=('self_attn_s.' + name,), blocks=[0])
                parameters = dict(model.named_parameters())
                if first is None:
                    first = {n: p for n, p in parameters.items() if '.lora_' in n}
                for key, parameter in first.items():
                    self.assertIs(parameters[key], parameter)
                for key, parameter in original.items():
                    self.assertIs(parameters[key], parameter)
            self.assertEqual(len([n for n, _ in model.named_parameters() if '.lora_' in n]), 8)

    def test_task_head_flags_dtype_and_existing_flags_without_freezing(self):
        for kind in (CBraModClassifier, CBraModRegression, CBraModEmbedding):
            model = task(kind).double()
            model.backbone.encoder.layers[0].linear1.weight.requires_grad_(False)
            original = {n: p.requires_grad for n, p in model.named_parameters()}
            add_cbramod_lora(model, rank=2, freeze_base=False, target_modules=TARGETS)
            for name, parameter in model.named_parameters():
                if name in original:
                    self.assertEqual(parameter.requires_grad, original[name])
                self.assertEqual(parameter.dtype, torch.float64)
            output = model(torch.randn(2, 3, 128, dtype=torch.float64))
            self.assertTrue(torch.isfinite(output).all())

    def test_dropout_modes_and_no_evaluation_weight_merging(self):
        model = encoder().eval()
        add_cbramod_lora(model, rank=2, dropout=.7, target_modules=TARGETS)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith('lora_B'):
                    parameter.normal_(std=.1)
        before = {n: v.clone() for n, v in model.state_dict().items()}
        x = torch.randn(2, 3, 4, 25)
        expected = model(x)
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        model.train()
        self.assertFalse(torch.equal(model(x), model(x)))
        model.eval()
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_masked_signal_gradient_isolation_with_attention_lora(self):
        model = encoder()
        add_cbramod_lora(model, rank=2, dropout=0., target_modules=TARGETS)
        x = torch.randn(2, 3, 4, 25, requires_grad=True)
        mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        mask[:, :, 0] = True
        output = model.forward_features(x, mask=mask)
        (output * torch.linspace(-1, 1, output.numel()).reshape(output.shape)).sum().backward()
        torch.testing.assert_close(x.grad[:, :, 0], torch.zeros_like(x.grad[:, :, 0]), rtol=0, atol=0)
        self.assertGreater(x.grad[:, :, 1:].norm().item(), 0)
        self.assertTrue(all(p.grad is not None for n, p in model.named_parameters() if '.lora_' in n))

    def test_invalid_or_duplicate_targets_fail_before_any_model_mutation(self):
        cases = [dict(rank=0), dict(rank=True), dict(alpha=0), dict(alpha=True), dict(alpha=float('inf')),
                 dict(alpha=float('nan')), dict(dropout=1.), dict(target_modules=[]),
                 dict(target_modules='linear1'), dict(target_modules=['linear1', 'linear1']),
                 dict(target_modules=['self_attn_s']), dict(target_modules=['self_attn_s.in_proj']),
                 dict(target_modules=[{}]), dict(blocks=[0, 0]), dict(freeze_base=1)]
        for settings in cases:
            model = encoder()
            original = dict(model.named_parameters())
            flags = {n: p.requires_grad for n, p in original.items()}
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                add_cbramod_lora(model, **settings)
            self.assertEqual(set(original), set(dict(model.named_parameters())))
            for name, parameter in model.named_parameters():
                self.assertIs(parameter, original[name])
                self.assertEqual(parameter.requires_grad, flags[name])
        model = encoder()
        add_cbramod_lora(model, blocks=[1], target_modules=('linear1',))
        with self.assertRaises(ValueError):
            add_cbramod_lora(model, target_modules=('linear1', 'linear2'))
        self.assertIs(type(model.encoder.layers[0].linear1), nn.Linear)
        model = encoder()
        model.encoder.layers[1].self_attn_t = nn.Identity()
        with self.assertRaises(ValueError):
            add_cbramod_lora(model, target_modules=TARGETS)
        self.assertIs(type(model.encoder.layers[0].linear1), nn.Linear)
        with self.assertRaises(ValueError):
            add_cbramod_lora(task(freeze_encoder=True))


if __name__ == '__main__':
    unittest.main()
