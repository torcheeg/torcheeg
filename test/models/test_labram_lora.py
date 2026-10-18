import copy
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from torcheeg.models import (
    LaBraMEncoder,
    LaBraMClassifier,
    LaBraMRegression,
    LaBraMEmbedding,
    add_labram_adapters,
    add_labram_prefix,
    add_labram_lora,
)
from torcheeg.models.transformer._tuning import _LoRALinear
from torcheeg.models.transformer.eegpt_tuning import _LoRALinear as EEGPTLoRALinear


def encoder(**kwargs):
    config = dict(
        patch_size=16, hid_channels=16, conv_channels=8, mlp_channels=32, depth=2, num_heads=4
    )
    config.update(kwargs)
    return LaBraMEncoder(**config)


def tensor(record):
    return torch.tensor(record['values'], dtype=torch.float64).reshape(record['shape'])


class TestLaBraMLoRA(unittest.TestCase):
    def test_independent_native_weight_parametrization_outputs_and_all_gradients(self):
        fixture = json.loads(
            (Path(__file__).parent / 'fixtures' / 'labram_lora_reference.json').read_text()
        )
        for case in fixture['cases']:
            with self.subTest(qkv_bias=case['qkv_bias'], qk_norm=case['qk_norm']):
                model = (
                    LaBraMEncoder(
                        **fixture['config'], qkv_bias=case['qkv_bias'], qk_norm=case['qk_norm']
                    )
                    .double()
                    .eval()
                )
                add_labram_lora(
                    model,
                    rank=fixture['rank'],
                    alpha=fixture['alpha'],
                    dropout=0.0,
                    freeze_base=False,
                )
                model.load_state_dict(
                    {n: tensor(v) for n, v in case['state_dict'].items()}, strict=True
                )
                x = tensor(case['input']).requires_grad_()
                actual = model.forward_all(x, torch.tensor(case['channel_indices']))
                torch.testing.assert_close(actual, tensor(case['output']), rtol=1e-10, atol=1e-11)
                (actual * tensor(case['coefficient'])).sum().backward()
                torch.testing.assert_close(
                    x.grad, tensor(case['input_gradient']), rtol=1e-9, atol=1e-10
                )
                self.assertEqual(
                    set(dict(model.named_parameters())), set(case['parameter_gradients'])
                )
                for name, p in model.named_parameters():
                    expected = case['parameter_gradients'][name]
                    if expected is None:
                        self.assertIsNone(p.grad, name)
                    else:
                        torch.testing.assert_close(
                            p.grad, tensor(expected), rtol=1e-9, atol=1e-10, msg=name
                        )

    def test_zero_insertion_identity_native_parameter_ownership_and_placement(self):
        self.assertIs(_LoRALinear, EEGPTLoRALinear)
        model = encoder(qkv_bias=True).double().eval()
        model.blocks[0].mlp.drop.train()
        x, indices = torch.randn(2, 2, 32, dtype=torch.float64), torch.tensor([1, 40])
        expected = model.forward_all(x, indices)
        before = dict(model.named_parameters())
        self.assertIs(add_labram_lora(model, rank=3, dropout=0.0), model)
        torch.testing.assert_close(model.forward_all(x, indices), expected, rtol=0, atol=0)
        after = dict(model.named_parameters())
        for n, p in before.items():
            self.assertIs(after[n], p)
            self.assertFalse(p.requires_grad)
        self.assertTrue(model.blocks[0].mlp.drop.training)
        layers = [m for m in model.modules() if isinstance(m, _LoRALinear)]
        self.assertEqual(len(layers), 8)
        self.assertEqual(len({id(m.lora_A) for m in layers}), 8)
        self.assertTrue(all(m.lora_A.dtype == torch.float64 and not m.training for m in layers))
        self.assertTrue(all(torch.count_nonzero(m.lora_B).item() == 0 for m in layers))
        half = encoder().half()
        add_labram_lora(half)
        self.assertTrue(all(p.dtype == torch.float16 for p in half.parameters()))

    def test_qkv_only_updates_participate_in_all_packed_projections(self):
        model = encoder(qkv_bias=True).double().eval()
        x, indices = torch.randn(2, 2, 32, dtype=torch.float64), torch.tensor([1, 40])
        before = model.forward_all(x, indices)
        add_labram_lora(model, rank=3, dropout=0.0, blocks=[0], target_modules=('attn.qkv',))
        linear = model.blocks[0].attn.qkv
        with torch.no_grad():
            linear.lora_B.normal_(std=0.2)
        output = model.forward_all(x, indices)
        self.assertFalse(torch.allclose(output, before, rtol=1e-9, atol=1e-10))
        (
            output
            * torch.linspace(-0.7, 1.1, output.numel(), dtype=output.dtype).reshape_as(output)
        ).sum().backward()
        for gradient in linear.lora_B.grad.chunk(3, dim=1):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().max().item(), 0)
        self.assertGreater(linear.lora_A.grad.abs().max().item(), 0)

    def test_registered_input_dropout_matches_explicit_equation_and_modes(self):
        model = encoder().double().eval()
        add_labram_lora(
            model, rank=3, alpha=5.0, dropout=0.5, blocks=[0], target_modules=('mlp.fc1',)
        )
        linear = model.blocks[0].mlp.fc1
        with torch.no_grad():
            linear.lora_B.normal_(std=0.2)
        x = torch.randn(2, 5, 16, dtype=torch.float64)
        expected = F.linear(x, linear.weight, linear.bias) + x @ linear.lora_A @ linear.lora_B * (
            5.0 / 3
        )
        torch.testing.assert_close(linear(x), expected)
        linear.train()
        torch.manual_seed(42)
        expected = F.linear(x, linear.weight, linear.bias) + F.dropout(
            x, p=0.5
        ) @ linear.lora_A @ linear.lora_B * (5.0 / 3)
        torch.manual_seed(42)
        torch.testing.assert_close(linear(x), expected)
        self.assertTrue(linear.dropout.training)
        linear.eval()
        self.assertFalse(linear.dropout.training)
        torch.testing.assert_close(linear(x), linear(x), rtol=0, atol=0)

    def test_task_wrappers_train_low_rank_and_heads_without_changing_native_state(self):
        for kind in (LaBraMClassifier, LaBraMRegression, LaBraMEmbedding):
            with self.subTest(kind=kind.__name__):
                model = kind(
                    encoder(), ['Fp1', 'C3'], chunk_size=32, sampling_rate=200, freeze_encoder=False
                )
                native = {n: t.clone() for n, t in model.backbone.state_dict().items()}
                add_labram_lora(model, rank=3, dropout=0.0)
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                optimizer = torch.optim.Adam(
                    [p for p in model.parameters() if p.requires_grad], lr=0.01
                )
                x = torch.randn(2, 2, 32, requires_grad=True)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    output = model(x)
                    coefficient = torch.linspace(-0.7, 1.1, output.numel()).reshape_as(output)
                    (output * coefficient).sum().backward()
                    self.assertTrue(torch.isfinite(x.grad).all())
                    self.assertGreater(x.grad.abs().max().item(), 0)
                    for n, p in model.backbone.named_parameters():
                        self.assertEqual(p.requires_grad, '.lora_' in n, n)
                        if p.requires_grad:
                            self.assertIsNotNone(p.grad, n)
                            self.assertTrue(torch.isfinite(p.grad).all(), n)
                        else:
                            self.assertIsNone(p.grad, n)
                    optimizer.step()
                for n, value in native.items():
                    torch.testing.assert_close(
                        model.backbone.state_dict()[n], value, rtol=0, atol=0
                    )
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        self.assertFalse(torch.equal(p, before[n]), n)

    def test_all_six_composition_orders_and_strict_restoration(self):
        for order in itertools.permutations(('adapter', 'prefix', 'lora')):
            model = encoder().double().eval()
            x, indices = torch.randn(2, 2, 32, dtype=torch.float64), torch.tensor([1, 40])

            def install(target, method):
                if method == 'adapter':
                    add_labram_adapters(target, adapter_size=4, dropout=0.0)
                elif method == 'prefix':
                    add_labram_prefix(target, prefix_length=3)
                else:
                    add_labram_lora(target, rank=3, dropout=0.0)

            for method in order:
                before = model.forward_all(x, indices)
                install(model, method)
                if method != 'prefix':
                    torch.testing.assert_close(
                        model.forward_all(x, indices), before, rtol=0, atol=0
                    )
            for n, p in model.named_parameters():
                self.assertEqual(
                    p.requires_grad,
                    '.adapter_' in n or '.lora_' in n or n.endswith('prefix_tokens'),
                    n,
                )
            other = encoder().double().eval()
            for method in order:
                install(other, method)
            other.load_state_dict(model.state_dict(), strict=True)
            torch.testing.assert_close(
                other.forward_all(x, indices), model.forward_all(x, indices), rtol=0, atol=0
            )
            clone = copy.deepcopy(model)
            torch.testing.assert_close(
                clone.forward_all(x, indices), model.forward_all(x, indices), rtol=0, atol=0
            )
            output = clone.forward_all(x.requires_grad_(), indices)
            (
                output
                * torch.linspace(-0.7, 1.1, output.numel(), dtype=output.dtype).reshape_as(output)
            ).sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            for n, p in clone.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, n)
                    self.assertTrue(torch.isfinite(p.grad).all(), n)

    def test_disjoint_targets_false_freeze_and_head_flags(self):
        model = LaBraMClassifier(encoder(), ['Fp1', 'C3'], chunk_size=32, sampling_rate=200)
        model.backbone.blocks[0].norm1.requires_grad_(False)
        model.task_head.requires_grad_(False)
        flags = {n: p.requires_grad for n, p in model.named_parameters()}
        add_labram_lora(model, rank=3, blocks=[0], target_modules=('attn.qkv',), freeze_base=False)
        for n, p in model.named_parameters():
            self.assertEqual(p.requires_grad, flags.get(n, True), n)
        add_labram_lora(model, rank=2, blocks=[0], target_modules=('attn.proj',))
        add_labram_lora(model, rank=4, blocks=[1], target_modules=('mlp.fc1', 'mlp.fc2'))
        for n, p in model.backbone.named_parameters():
            self.assertEqual(p.requires_grad, '.lora_' in n, n)
        self.assertTrue(all(not p.requires_grad for p in model.task_head.parameters()))
        self.assertEqual(model.backbone.blocks[0].attn.qkv.lora_A.shape[1], 3)
        self.assertEqual(model.backbone.blocks[0].attn.proj.lora_A.shape[1], 2)

    def test_invalid_selection_is_atomic_and_native_loader_rejects_augmented_state(self):
        invalid = [dict(rank=v) for v in (0, -1, True, 1.5)]
        invalid += [dict(alpha=v) for v in (0, -1, True, float('nan'), float('inf'))]
        invalid += [dict(dropout=v) for v in (-1, 1, True, float('nan'))]
        invalid += [
            dict(target_modules=v)
            for v in (
                [],
                ['qkv'],
                ['attn.qkv', 'attn.qkv'],
                ['task_head'],
                ['mlp.fc1', 'adapter_attn.up_project'],
            )
        ]
        invalid += [dict(blocks=[0, 0]), dict(blocks=[2]), dict(freeze_base='yes')]
        for kwargs in invalid:
            model = encoder()
            before = dict(model.named_parameters())
            with self.assertRaises(ValueError):
                add_labram_lora(model, **kwargs)
            self.assertEqual(dict(model.named_parameters()), before)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
        model = encoder()
        add_labram_lora(model, blocks=[1], target_modules=('attn.qkv',))
        with self.assertRaises(ValueError):
            add_labram_lora(model, blocks=[0, 1], target_modules=('mlp.fc1', 'attn.qkv'))
        self.assertIs(type(model.blocks[0].mlp.fc1), nn.Linear)
        self.assertIs(type(model.blocks[0].attn.qkv), nn.Linear)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'augmented.pt'
            torch.save(model.state_dict(), path)
            with self.assertRaises(ValueError):
                LaBraMEncoder.from_pretrained(
                    path,
                    patch_size=16,
                    hid_channels=16,
                    conv_channels=8,
                    mlp_channels=32,
                    depth=2,
                    num_heads=4,
                )
        task = LaBraMClassifier(
            encoder(), ['Fp1', 'C3'], chunk_size=32, sampling_rate=200, freeze_encoder=True
        )
        with self.assertRaises(ValueError):
            add_labram_lora(task)

    @unittest.skipUnless(
        os.environ.get('GITHUB_ACTIONS') == 'true'
        or os.environ.get('TORCHEEG_TEST_COMPILE') == '1',
        'Optional full-graph compiler regression',
    )
    def test_fullgraph_training_with_nonzero_lora_and_all_tuning_modules(self):
        for qkv_bias, qk_norm in ((False, True), (True, False)):
            model = encoder(qkv_bias=qkv_bias, qk_norm=qk_norm).eval()
            add_labram_lora(model, rank=3, dropout=0.0)
            add_labram_prefix(model, prefix_length=3)
            add_labram_adapters(model, adapter_size=4, dropout=0.0)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n.endswith('lora_B') or n.endswith('up_project.weight'):
                        p.normal_(std=0.1)
            reference = copy.deepcopy(model)
            x = torch.randn(2, 2, 32, requires_grad=True)
            other = x.detach().clone().requires_grad_()
            indices = torch.tensor([1, 40])

            def forward(value):
                return model.forward_all(value, indices)

            actual = torch.compile(forward, fullgraph=True, dynamic=False)(x)
            expected = reference.forward_all(other, indices)
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
            coefficient = torch.linspace(-0.7, 1.1, actual.numel()).reshape_as(actual)
            (actual * coefficient).sum().backward()
            (expected * coefficient).sum().backward()
            torch.testing.assert_close(x.grad, other.grad, rtol=3e-3, atol=1e-4)
            for (n, p), (m, q) in zip(model.named_parameters(), reference.named_parameters()):
                self.assertEqual(n, m)
                if q.grad is None:
                    self.assertIsNone(p.grad, n)
                else:
                    torch.testing.assert_close(p.grad, q.grad, rtol=3e-3, atol=1e-4, msg=n)


if __name__ == '__main__':
    unittest.main()
