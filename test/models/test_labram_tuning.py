import copy
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
)
from torcheeg.models.transformer import add_labram_adapters as TransformerImport
from torcheeg.models.transformer._tuning import _Adapter
from torcheeg.models.transformer.eegpt_tuning import _Adapter as EEGPTAdapter


CONFIG = dict(
    patch_size=16, hid_channels=16, conv_channels=8, mlp_channels=32, depth=2, num_heads=4
)
NAMES = ['FP1', 'C3']


def encoder(**kwargs):
    return LaBraMEncoder(**dict(CONFIG, **kwargs))


def tensor(record):
    return torch.tensor(record['values'], dtype=torch.float64).reshape(record['shape'])


class TestLaBraMAdapters(unittest.TestCase):
    def test_zero_insertion_preserves_native_objects_scales_modes_and_dtype(self):
        self.assertIs(add_labram_adapters, TransformerImport)
        self.assertIs(_Adapter, EEGPTAdapter)
        for scale in [0.0, 0.1]:
            model = encoder(layer_scale=scale).double().eval()
            before = dict(model.named_parameters())
            attention = model.blocks[0].attn
            gamma = model.blocks[0].gamma_1
            x = torch.randn(2, 2, 32, dtype=torch.float64)
            indices = model.channel_indices(NAMES)
            expected = model.forward_all(x, indices)
            self.assertIs(add_labram_adapters(model, adapter_size=4), model)
            torch.testing.assert_close(model.forward_all(x, indices), expected, rtol=0, atol=0)
            self.assertIs(model.blocks[0].attn, attention)
            self.assertIs(model.blocks[0].gamma_1, gamma)
            after = dict(model.named_parameters())
            for name, parameter in before.items():
                self.assertIs(after[name], parameter)
                self.assertFalse(parameter.requires_grad)
            self.assertTrue(all(not module.training for module in model.modules()))
            self.assertTrue(all(p.dtype == torch.float64 for p in after.values()))
            self.assertTrue(all(p.requires_grad for n, p in after.items() if '.adapter_' in n))
            for block in model.blocks:
                self.assertEqual(
                    torch.count_nonzero(block.adapter_attn.up_project.weight).item(), 0
                )
                self.assertEqual(torch.count_nonzero(block.adapter_mlp.up_project.weight).item(), 0)

    def test_independent_archived_adapter_native_encoder_outputs_and_all_gradients(self):
        reference = json.loads(
            (Path(__file__).parent / 'fixtures' / 'labram_adapter_reference.json').read_text()
        )
        model = LaBraMEncoder(**reference['config']).double().eval()
        add_labram_adapters(
            model, adapter_size=reference['adapter_size'], dropout=0.0, freeze_base=False
        )
        model.load_state_dict(
            {n: tensor(v) for n, v in reference['state_dict'].items()}, strict=True
        )
        x = tensor(reference['input']).requires_grad_()
        actual = model.forward_all(x, torch.tensor(reference['channel_indices']))
        torch.testing.assert_close(actual, tensor(reference['output']), rtol=1e-10, atol=1e-11)
        (actual * tensor(reference['coefficient'])).sum().backward()
        torch.testing.assert_close(
            x.grad, tensor(reference['input_gradient']), rtol=1e-9, atol=1e-10
        )
        self.assertEqual(set(dict(model.named_parameters())), set(reference['parameter_gradients']))
        for name, p in model.named_parameters():
            expected = reference['parameter_gradients'][name]
            if expected is None:
                self.assertIsNone(p.grad, name)
            else:
                torch.testing.assert_close(
                    p.grad, tensor(expected), rtol=1e-9, atol=1e-10, msg=name
                )

    def test_active_adapters_with_layer_scales_disabled_follow_both_residuals(self):
        model = encoder(layer_scale=0.0).double().eval()
        add_labram_adapters(model, adapter_size=4, dropout=0.0, blocks=[0])
        block = model.blocks[0]
        with torch.no_grad():
            block.adapter_attn.up_project.weight.normal_(std=0.2)
            block.adapter_mlp.up_project.weight.normal_(std=0.2)
        x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_()

        def adapter(value, module):
            return value + F.linear(
                F.gelu(F.linear(value, module.down_project.weight, module.down_project.bias)),
                module.up_project.weight,
                module.up_project.bias,
            )

        first = adapter(y + block.attn(block.norm1(y)), block.adapter_attn)
        expected = adapter(first + block.mlp(block.norm2(first)), block.adapter_mlp)
        actual = block(x)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        coefficient = torch.linspace(-0.7, 1.1, actual.numel(), dtype=torch.float64).reshape_as(
            actual
        )
        params = list(block.adapter_attn.parameters()) + list(block.adapter_mlp.parameters())
        left = torch.autograd.grad((actual * coefficient).sum(), [x] + params)
        right = torch.autograd.grad((expected * coefficient).sum(), [y] + params)
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)

    def test_selected_blocks_have_independent_parameters_and_retain_installed_adapters(self):
        model = encoder()
        add_labram_adapters(model, adapter_size=4, blocks=[1])
        self.assertFalse(hasattr(model.blocks[0], 'adapter_attn'))
        first = model.blocks[1].adapter_attn
        self.assertEqual(sum(p.numel() for p in first.parameters()), 16 * 4 + 4 + 4 * 16 + 16)
        first.requires_grad_(False)
        add_labram_adapters(model, adapter_size=3, blocks=[0])
        second = model.blocks[0].adapter_attn
        self.assertIsNot(first, second)
        self.assertNotEqual(first.up_project.weight.data_ptr(), second.up_project.weight.data_ptr())
        self.assertTrue(all(p.requires_grad for p in first.parameters()))
        self.assertTrue(all(p.requires_grad for p in second.parameters()))
        self.assertIsNot(model.blocks[0].adapter_attn, model.blocks[0].adapter_mlp)

    def test_head_and_existing_parameter_flags_are_preserved_without_base_freezing(self):
        for kind in (LaBraMClassifier, LaBraMRegression, LaBraMEmbedding):
            model = kind(encoder(), NAMES, 32)
            model.backbone.cls_token.requires_grad_(False)
            if hasattr(model, 'task_head'):
                model.task_head.output.bias.requires_grad_(False)
            before = {n: p.requires_grad for n, p in model.named_parameters()}
            add_labram_adapters(model, adapter_size=4, freeze_base=False)
            after = dict(model.named_parameters())
            for name, flag in before.items():
                self.assertEqual(after[name].requires_grad, flag, name)
            self.assertTrue(model.backbone.training)
            self.assertTrue(model.backbone.blocks[0].adapter_attn.training)

    def test_two_steps_update_adapters_and_head_without_changing_any_base_tensor(self):
        for kind in (LaBraMClassifier, LaBraMRegression, LaBraMEmbedding):
            with self.subTest(kind=kind.__name__):
                model = kind(encoder(), NAMES, 32)
                before = {n: v.clone() for n, v in model.backbone.state_dict().items()}
                head = (
                    model.task_head.output.weight.detach().clone()
                    if hasattr(model, 'task_head')
                    else None
                )
                add_labram_adapters(model, adapter_size=4, dropout=0.0)
                optimizer = torch.optim.Adam(
                    [p for p in model.parameters() if p.requires_grad], lr=0.01
                )
                x = torch.randn(2, 2, 32, requires_grad=True)
                for step in range(2):
                    optimizer.zero_grad()
                    x.grad = None
                    output = model(x)
                    if kind == LaBraMClassifier:
                        loss = F.cross_entropy(output, torch.tensor([0, 1]))
                    elif kind == LaBraMRegression:
                        loss = F.l1_loss(output, torch.tensor([[0.3], [-0.5]]))
                    else:
                        loss = (
                            output * torch.linspace(-0.7, 1.1, output.numel()).reshape_as(output)
                        ).sum()
                    loss.backward()
                    self.assertTrue(torch.isfinite(x.grad).all())
                    self.assertGreater(x.grad.abs().max().item(), 0.0)
                    for block in model.backbone.blocks:
                        for adapter in [block.adapter_attn, block.adapter_mlp]:
                            self.assertGreater(
                                adapter.up_project.weight.grad.abs().max().item(), 0.0
                            )
                            if step:
                                self.assertGreater(
                                    adapter.down_project.weight.grad.abs().max().item(), 0.0
                                )
                    optimizer.step()
                if head is not None:
                    self.assertFalse(torch.equal(head, model.task_head.output.weight))
                for name, value in model.backbone.state_dict().items():
                    if name in before:
                        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
                self.assertTrue(
                    all(
                        p.grad is None
                        for n, p in model.backbone.named_parameters()
                        if '.adapter_' not in n
                    )
                )

    def test_copy_strict_augmented_state_and_parameter_independence(self):
        model = encoder().double().eval()
        add_labram_adapters(model, adapter_size=4, blocks=[1], dropout=0.0)
        with torch.no_grad():
            model.blocks[1].adapter_attn.up_project.weight.normal_(std=0.1)
            model.blocks[1].adapter_mlp.up_project.weight.normal_(std=0.1)
        x = torch.randn(2, 2, 32, dtype=torch.float64)
        indices = model.channel_indices(NAMES)
        expected = model(x, indices)
        copied = copy.deepcopy(model)
        torch.testing.assert_close(copied(x, indices), expected, rtol=0, atol=0)
        self.assertNotEqual(
            copied.blocks[1].adapter_mlp.up_project.weight.data_ptr(),
            model.blocks[1].adapter_mlp.up_project.weight.data_ptr(),
        )
        with torch.no_grad():
            copied.blocks[1].adapter_mlp.up_project.bias[0].add_(1)
        self.assertFalse(torch.equal(copied(x, indices), model(x, indices)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'augmented.pt'
            torch.save(model.state_dict(), path)
            restored = encoder().double().eval()
            add_labram_adapters(restored, adapter_size=4, blocks=[1], dropout=0.0)
            restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
            torch.testing.assert_close(restored(x, indices), expected, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                LaBraMEncoder.from_pretrained(path, **CONFIG)

    def test_dropout_and_mixed_modes_are_registered_and_preserved_at_insertion(self):
        model = encoder().double().train()
        model.blocks[0].eval()
        model.blocks[0].mlp.train()
        mlp = model.blocks[0].mlp
        add_labram_adapters(model, adapter_size=4, dropout=0.7)
        self.assertIs(model.blocks[0].mlp, mlp)
        self.assertTrue(mlp.training)
        self.assertFalse(model.blocks[0].adapter_attn.training)
        self.assertTrue(model.blocks[1].adapter_attn.training)
        adapter = model.blocks[0].adapter_attn
        with torch.no_grad():
            adapter.up_project.weight.normal_(std=0.1)
        x = torch.randn(2, 5, 16, dtype=torch.float64)
        torch.testing.assert_close(adapter(x), adapter(x), rtol=0, atol=0)
        model.train()
        self.assertFalse(torch.equal(adapter(x), adapter(x)))
        model.float().eval()
        self.assertTrue(all(p.dtype == torch.float32 for p in adapter.parameters()))

    def test_invalid_duplicate_custom_or_frozen_task_selection_is_atomic(self):
        for kwargs in [
            dict(adapter_size=0),
            dict(adapter_size=True),
            dict(dropout=1.0),
            dict(dropout=float('nan')),
            dict(blocks=[]),
            dict(blocks=[0, 0]),
            dict(blocks=[-1]),
            dict(blocks=[2]),
            dict(blocks=[False]),
            dict(blocks=[0.5]),
            dict(freeze_base=1),
        ]:
            model = encoder()
            before = dict(model.named_parameters())
            flags = {n: p.requires_grad for n, p in before.items()}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                add_labram_adapters(model, **kwargs)
            self.assertEqual(set(before), set(dict(model.named_parameters())))
            self.assertEqual(flags, {n: p.requires_grad for n, p in model.named_parameters()})
        model = encoder()
        add_labram_adapters(model, blocks=[0])
        with self.assertRaises(ValueError):
            add_labram_adapters(model)
        self.assertFalse(hasattr(model.blocks[1], 'adapter_attn'))
        for kind in (LaBraMClassifier, LaBraMRegression, LaBraMEmbedding):
            with self.assertRaises(ValueError):
                add_labram_adapters(kind(encoder(), NAMES, 32, freeze_encoder=True))
        with self.assertRaises(ValueError):
            add_labram_adapters(nn.Identity())
        model = encoder()
        model.blocks[1] = nn.Identity()
        with self.assertRaises(ValueError):
            add_labram_adapters(model)
        self.assertFalse(hasattr(model.blocks[0], 'adapter_attn'))

    @unittest.skipUnless(
        os.environ.get('GITHUB_ACTIONS') == 'true'
        or os.environ.get('TORCHEEG_TEST_COMPILE') == '1',
        'Optional compiler regression',
    )
    def test_compiled_layer_scale_and_unscaled_adapters_match_all_gradients(self):
        for scale in [0.1, 0.0]:
            reference = LaBraMClassifier(
                encoder(layer_scale=scale), NAMES, 48, sampling_rate=150, dropout=0.0
            ).eval()
            add_labram_adapters(reference, adapter_size=4, dropout=0.0, freeze_base=False)
            with torch.no_grad():
                for block in reference.backbone.blocks:
                    block.adapter_attn.up_project.weight.normal_(std=0.1)
                    block.adapter_mlp.up_project.weight.normal_(std=0.1)
            candidate = copy.deepcopy(reference)
            x = torch.randn(2, 2, 48, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            expected = reference(x)
            actual = torch.compile(candidate, backend='inductor', fullgraph=True, dynamic=False)(y)
            coefficient = torch.tensor([[0.2, -0.3], [-0.4, 0.7]])
            (expected * coefficient).sum().backward()
            (actual * coefficient).sum().backward()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(y.grad, x.grad, rtol=2e-3, atol=1e-5)
            for (name, left), (_, right) in zip(
                candidate.named_parameters(), reference.named_parameters()
            ):
                torch.testing.assert_close(left.grad, right.grad, rtol=2e-3, atol=1e-5, msg=name)


if __name__ == '__main__':
    unittest.main()
