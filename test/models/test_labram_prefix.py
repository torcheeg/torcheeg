import copy
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from torcheeg.models import (
    LaBraMEncoder,
    LaBraMClassifier,
    LaBraMRegression,
    LaBraMEmbedding,
    add_labram_adapters,
    add_labram_prefix,
)
from torcheeg.models.transformer.labram_encoder import _Attention


def encoder(**kwargs):
    config = dict(
        patch_size=16, hid_channels=16, conv_channels=8, mlp_channels=32, depth=2, num_heads=4
    )
    config.update(kwargs)
    return LaBraMEncoder(**config)


def tensor(record):
    return torch.tensor(record['values'], dtype=torch.float64).reshape(record['shape'])


class TestLaBraMPrefix(unittest.TestCase):
    def test_archived_prefix_reference_outputs_and_every_gradient(self):
        fixture = json.loads(
            (Path(__file__).parent / 'fixtures' / 'labram_prefix_reference.json').read_text()
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
                add_labram_prefix(model, fixture['prefix_length'], freeze_base=False)
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

    def test_preserves_native_ownership_paths_mixed_modes_and_dtype(self):
        model = encoder(qkv_bias=True).double().eval()
        model.blocks[0].attn.attn_drop.train()
        model.blocks[0].attn.q_norm.train()
        modules = {n: m for n, m in model.named_modules()}
        parameters = dict(model.named_parameters())
        modes = {n: m.training for n, m in modules.items()}
        self.assertIs(add_labram_prefix(model, prefix_length=3), model)
        after = dict(model.named_modules())
        for name, module in modules.items():
            if name not in ('blocks.0.attn', 'blocks.1.attn'):
                self.assertIs(after[name], module, name)
            self.assertEqual(after[name].training, modes[name], name)
        for name, p in parameters.items():
            self.assertIs(dict(model.named_parameters())[name], p)
            self.assertFalse(p.requires_grad)
        for block in model.blocks:
            self.assertEqual(tuple(block.attn.prefix_tokens.shape), (3, 2, 4, 4))
            self.assertEqual(block.attn.prefix_tokens.dtype, torch.float64)
            self.assertTrue(block.attn.prefix_tokens.requires_grad)
        self.assertIsNot(model.blocks[0].attn.prefix_tokens, model.blocks[1].attn.prefix_tokens)
        self.assertNotIn('blocks.0.attn.prefix_tokens', parameters)

    def test_prefix_is_not_input_tokens_and_zero_prefix_is_not_identity(self):
        model = encoder().double().eval()
        x, indices = torch.randn(2, 2, 32, dtype=torch.float64), torch.tensor([1, 40])
        original = model.forward_all(x, indices)
        add_labram_prefix(model, prefix_length=3)
        for block in model.blocks:
            nn.init.zeros_(block.attn.prefix_tokens)
        output = model.forward_all(x, indices)
        self.assertEqual(output.shape, original.shape)
        self.assertEqual(output.shape[1], 5)
        self.assertFalse(torch.allclose(output, original, rtol=1e-9, atol=1e-10))

    def test_all_wrappers_two_updates_freeze_native_and_keep_input_gradients(self):
        for kind in (LaBraMClassifier, LaBraMRegression, LaBraMEmbedding):
            with self.subTest(kind=kind.__name__):
                model = kind(
                    encoder(), ['Fp1', 'C3'], chunk_size=32, sampling_rate=200, freeze_encoder=False
                )
                if kind is not LaBraMEmbedding:
                    model.task_head.requires_grad_(True)
                native = {n: t.clone() for n, t in model.backbone.state_dict().items()}
                head_flags = {
                    n: p.requires_grad
                    for n, p in model.named_parameters()
                    if not n.startswith('backbone.')
                }
                add_labram_prefix(model, prefix_length=3)
                before = {
                    n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
                }
                optimizer = torch.optim.Adam(
                    [p for p in model.parameters() if p.requires_grad], lr=0.01
                )
                x = torch.randn(2, 2, 32, requires_grad=True)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    output = model(x)
                    coefficient = torch.linspace(-0.7, 1.1, output.numel()).reshape_as(output)
                    (output * coefficient).sum().backward()
                    for n, p in model.backbone.named_parameters():
                        if n.endswith('prefix_tokens'):
                            self.assertTrue(torch.isfinite(p.grad).all())
                            self.assertGreater(p.grad[:, 0].abs().max().item(), 0)
                            self.assertGreater(p.grad[:, 1].abs().max().item(), 0)
                        else:
                            self.assertIsNone(p.grad, n)
                    self.assertTrue(torch.isfinite(x.grad).all())
                    self.assertGreater(x.grad.abs().max().item(), 0)
                    optimizer.step()
                for n, t in native.items():
                    torch.testing.assert_close(model.backbone.state_dict()[n], t, rtol=0, atol=0)
                for n, p in model.named_parameters():
                    if n.endswith('prefix_tokens'):
                        self.assertFalse(torch.equal(before[n], p), n)
                    elif n in head_flags:
                        self.assertEqual(p.requires_grad, head_flags[n])

    def test_adapter_composition_both_orders_and_disjoint_installations(self):
        for order in itertools.permutations(('adapter', 'prefix')):
            model = encoder().double().eval()
            for method in order:
                if method == 'adapter':
                    add_labram_adapters(model, adapter_size=4, dropout=0.0)
                else:
                    add_labram_prefix(model, prefix_length=3, blocks=[0])
                    add_labram_prefix(model, prefix_length=2, blocks=[1])
            for n, p in model.named_parameters():
                self.assertEqual(
                    p.requires_grad, '.adapter_' in n or n.endswith('prefix_tokens'), n
                )
            x = torch.randn(2, 2, 32, dtype=torch.float64, requires_grad=True)
            y = model.forward_all(x, torch.tensor([1, 40]))
            (y * torch.linspace(-0.7, 1.1, y.numel(), dtype=y.dtype).reshape_as(y)).sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, n)
                    self.assertTrue(torch.isfinite(p.grad).all(), n)
            restored = copy.deepcopy(model)
            restored.load_state_dict(model.state_dict(), strict=True)
            torch.testing.assert_close(
                restored.forward_all(x, torch.tensor([1, 40])), y, rtol=0, atol=0
            )

    def test_false_freeze_preserves_flags_and_task_head_flags(self):
        model = LaBraMClassifier(encoder(), ['Fp1', 'C3'], chunk_size=32, sampling_rate=200)
        model.backbone.blocks[0].mlp.fc1.weight.requires_grad_(False)
        model.task_head.requires_grad_(False)
        flags = {n: p.requires_grad for n, p in model.named_parameters()}
        add_labram_prefix(model, prefix_length=2, blocks=[1], freeze_base=False)
        for n, p in model.named_parameters():
            self.assertEqual(p.requires_grad, flags.get(n, True), n)

    def test_strict_restoration_requires_matching_prefix_structure(self):
        model = encoder().eval()
        add_labram_prefix(model, prefix_length=3)
        restored = encoder().eval()
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(model.state_dict(), strict=True)
        add_labram_prefix(restored, prefix_length=3)
        restored.load_state_dict(model.state_dict(), strict=True)
        x, indices = torch.randn(2, 2, 32), torch.tensor([1, 40])
        torch.testing.assert_close(restored(x, indices), model(x, indices), rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'augmented.pt'
            torch.save(model.state_dict(), checkpoint)
            with self.assertRaises(ValueError):
                LaBraMEncoder.from_pretrained(
                    checkpoint,
                    patch_size=16,
                    hid_channels=16,
                    conv_channels=8,
                    mlp_channels=32,
                    depth=2,
                    num_heads=4,
                )

    def test_invalid_arguments_and_duplicate_or_custom_selection_are_atomic(self):
        invalid = [dict(prefix_length=v) for v in (0, -1, True, 1.5)]
        invalid += [dict(blocks=v) for v in ([], [0, 0], [2], [-1], [True], 'all')]
        invalid += [dict(freeze_base='yes')]
        for kwargs in invalid:
            model = encoder()
            parameters = dict(model.named_parameters())
            with self.assertRaises(ValueError):
                add_labram_prefix(model, **kwargs)
            self.assertEqual(dict(model.named_parameters()), parameters)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
        model = encoder()
        add_labram_prefix(model, blocks=[1])
        before = list(model.blocks)
        with self.assertRaises(ValueError):
            add_labram_prefix(model, blocks=[0, 1])
        self.assertEqual(list(model.blocks), before)
        self.assertIs(type(model.blocks[0].attn), _Attention)

        class CustomAttention(_Attention):
            pass

        model.blocks[0].attn.__class__ = CustomAttention
        with self.assertRaises(ValueError):
            add_labram_prefix(model, blocks=[0])
        task = LaBraMClassifier(
            encoder(), ['Fp1', 'C3'], chunk_size=32, sampling_rate=200, freeze_encoder=True
        )
        with self.assertRaises(ValueError):
            add_labram_prefix(task)
        with self.assertRaises(ValueError):
            add_labram_prefix(nn.Linear(2, 2))

    def test_attention_dropout_modes_cpu_bfloat16_and_native_half_placement(self):
        model = encoder(attention_dropout=0.5).eval()
        add_labram_prefix(model, prefix_length=3)
        x = torch.randn(2, 5, 16)
        attention = model.blocks[0].attn
        torch.testing.assert_close(attention(x), attention(x), rtol=0, atol=0)
        attention.train()
        self.assertFalse(torch.equal(attention(x), attention(x)))
        model.eval()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = model.forward_all(torch.randn(2, 2, 32), torch.tensor([1, 40]))
        self.assertTrue(torch.isfinite(output).all())
        half = encoder().half()
        add_labram_prefix(half)
        self.assertTrue(all(p.dtype == torch.float16 for p in half.parameters()))

    @unittest.skipUnless(
        os.environ.get('GITHUB_ACTIONS') == 'true'
        or os.environ.get('TORCHEEG_TEST_COMPILE') == '1',
        'Optional full-graph compiler regression',
    )
    def test_fullgraph_training_matches_eager_with_adapters_and_prefixes(self):
        for qkv_bias, qk_norm in ((False, True), (True, False)):
            model = encoder(qkv_bias=qkv_bias, qk_norm=qk_norm).eval()
            add_labram_adapters(model, adapter_size=4, dropout=0.0)
            add_labram_prefix(model, prefix_length=3)
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
