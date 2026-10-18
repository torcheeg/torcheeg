import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from torcheeg.models import (CBraMod, CBraModClassifier, CBraModRegression,
                             CBraModEmbedding, add_cbramod_adapters)
from torcheeg.models.transformer import add_cbramod_adapters as TransformerImport
from torcheeg.models.transformer._tuning import _Adapter
from torcheeg.models.transformer.eegpt_tuning import _Adapter as EEGPTAdapter


CONFIG = dict(patch_size=25, hid_channels=16, mlp_channels=32, depth=2,
              num_heads=4, dropout=0., position_kernel=(3, 3))


def encoder():
    return CBraMod(**CONFIG)


class TestCBraModAdapters(unittest.TestCase):
    def test_public_import_zero_insertion_base_ownership_and_modes(self):
        self.assertIs(add_cbramod_adapters, TransformerImport)
        self.assertIs(_Adapter, EEGPTAdapter)
        model = encoder().double().eval()
        original = dict(model.named_parameters())
        components = {name: module for name, module in model.named_modules() if name.endswith('self_attn_s')}
        x = torch.randn(2, 3, 4, 25, dtype=torch.float64)
        expected = model(x)
        self.assertIs(add_cbramod_adapters(model, adapter_size=4), model)
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        after = dict(model.named_parameters())
        for name, parameter in original.items():
            self.assertIs(after[name], parameter)
            self.assertFalse(parameter.requires_grad)
        self.assertTrue(all(not m.training for m in model.modules()))
        for name, module in components.items():
            self.assertIs(dict(model.named_modules())[name], module)
        self.assertTrue(all(p.dtype == torch.float64 for p in model.parameters()))
        self.assertTrue(all(p.requires_grad for name, p in after.items() if '.adapter.' in name))
        self.assertTrue(all(torch.count_nonzero(block.adapter.up_project.weight) == 0 for block in model.encoder.layers))

    def test_active_post_block_adapter_matches_functional_output_and_gradients(self):
        model = encoder().double().eval()
        base = copy.deepcopy(model.encoder.layers[0])
        add_cbramod_adapters(model, adapter_size=4, dropout=0., blocks=[0])
        block = model.encoder.layers[0]
        with torch.no_grad():
            block.adapter.up_project.weight.copy_(torch.linspace(-.2, .3, 64).reshape(16, 4))
        x = torch.randn(2, 3, 4, 16, dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        base.requires_grad_(False)
        value = base(y)
        hidden = F.gelu(F.linear(value, block.adapter.down_project.weight, block.adapter.down_project.bias))
        expected = value + F.linear(hidden, block.adapter.up_project.weight, block.adapter.up_project.bias)
        actual = block(x)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        coefficients = torch.linspace(-1, 1, actual.numel(), dtype=torch.float64).reshape(actual.shape)
        parameters = list(block.adapter.parameters())
        first = torch.autograd.grad((actual * coefficients).sum(), [x] + parameters)
        second = torch.autograd.grad((expected * coefficients).sum(), [y] + parameters)
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-12)
            self.assertTrue(torch.isfinite(left).all())

    def test_selected_blocks_independent_parameters_and_count(self):
        model = encoder()
        add_cbramod_adapters(model, adapter_size=4, blocks=[1])
        self.assertFalse(hasattr(model.encoder.layers[0], 'adapter'))
        first = model.encoder.layers[1].adapter
        self.assertEqual(sum(p.numel() for p in first.parameters()), 16 * 4 + 4 + 4 * 16 + 16)
        add_cbramod_adapters(model, adapter_size=3, blocks=[0])
        second = model.encoder.layers[0].adapter
        self.assertIsNot(first, second)
        self.assertNotEqual(first.up_project.weight.data_ptr(), second.up_project.weight.data_ptr())
        self.assertTrue(all(p.requires_grad for p in first.parameters()))
        self.assertTrue(all(p.requires_grad for p in second.parameters()))

    def test_task_head_flags_and_existing_base_flags_are_preserved(self):
        for kind in (CBraModClassifier, CBraModRegression, CBraModEmbedding):
            with self.subTest(kind=kind.__name__):
                model = kind(encoder(), 3, 128, num_patches=4)
                if hasattr(model, 'task_head'):
                    model.task_head.output.bias.requires_grad_(False)
                head_flags = {n: p.requires_grad for n, p in model.named_parameters() if n.startswith('task_head.')}
                original = {n: p.requires_grad for n, p in model.backbone.named_parameters()}
                add_cbramod_adapters(model, adapter_size=4, freeze_base=False)
                self.assertEqual(head_flags, {n: p.requires_grad for n, p in model.named_parameters() if n.startswith('task_head.')})
                self.assertEqual(original, {n: p.requires_grad for n, p in model.backbone.named_parameters() if '.adapter.' not in n})

    def test_two_optimizer_steps_update_adapters_head_and_no_base_state(self):
        model = CBraModClassifier(encoder(), 3, 128, num_patches=4, dropout=0.)
        before = {n: v.clone() for n, v in model.backbone.state_dict().items()}
        add_cbramod_adapters(model, adapter_size=4, dropout=0.)
        head = model.task_head.output.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        x = torch.randn(2, 3, 128, requires_grad=True)
        for step in range(2):
            optimizer.zero_grad()
            output = model(x)
            F.cross_entropy(output, torch.tensor([0, 1])).backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.norm().item(), 0)
            for block in model.backbone.encoder.layers:
                self.assertIsNotNone(block.adapter.up_project.weight.grad)
                self.assertGreater(block.adapter.up_project.weight.grad.norm().item(), 0)
                if step:
                    self.assertGreater(block.adapter.down_project.weight.grad.norm().item(), 0)
            optimizer.step()
        self.assertFalse(torch.equal(head, model.task_head.output.weight))
        for name, value in model.backbone.state_dict().items():
            if name in before:
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for n, p in model.backbone.named_parameters() if '.adapter.' not in n))

    def test_copy_state_restore_and_independent_module_ownership(self):
        model = encoder().double().eval()
        add_cbramod_adapters(model, adapter_size=4, dropout=0.)
        with torch.no_grad():
            for block in model.encoder.layers:
                block.adapter.up_project.weight.normal_(std=.1)
        copied = copy.deepcopy(model)
        x = torch.randn(2, 3, 4, 25, dtype=torch.float64)
        expected = model(x)
        torch.testing.assert_close(copied(x), expected, rtol=0, atol=0)
        self.assertNotEqual(copied.encoder.layers[0].adapter.up_project.weight.data_ptr(), model.encoder.layers[0].adapter.up_project.weight.data_ptr())
        with torch.no_grad():
            copied.encoder.layers[0].adapter.up_project.bias.add_(1)
        self.assertFalse(torch.equal(copied(x), model(x)))
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'adapters.pt'
            torch.save(model.state_dict(), path)
            restored = encoder().double().eval()
            add_cbramod_adapters(restored, adapter_size=4, dropout=0.)
            restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
            torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)

    def test_dropout_train_eval_and_dtype_moves(self):
        model = encoder().eval()
        add_cbramod_adapters(model, adapter_size=4, dropout=.7)
        adapter = model.encoder.layers[0].adapter
        with torch.no_grad():
            adapter.up_project.weight.normal_(std=.1)
        x = torch.randn(2, 3, 4, 16)
        torch.testing.assert_close(adapter(x), adapter(x), rtol=0, atol=0)
        model.train()
        self.assertTrue(adapter.training)
        self.assertFalse(torch.equal(adapter(x), adapter(x)))
        model.double().eval()
        self.assertTrue(all(p.dtype == torch.float64 for p in adapter.parameters()))
        self.assertEqual(adapter(x.double()).dtype, torch.float64)

    def test_masked_input_path_and_shared_parameter_state(self):
        model = encoder().eval()
        add_cbramod_adapters(model, adapter_size=4, dropout=0.)
        x = torch.randn(2, 3, 4, 25, requires_grad=True)
        mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        mask[:, :, 0] = True
        output = model.forward_features(x, mask=mask)
        (output * torch.linspace(-1, 1, output.numel()).reshape(output.shape)).sum().backward()
        torch.testing.assert_close(x.grad[:, :, 0], torch.zeros_like(x.grad[:, :, 0]), rtol=0, atol=0)
        self.assertGreater(x.grad[:, :, 1:].norm().item(), 0)

    def test_invalid_or_duplicate_selection_does_not_mutate_base(self):
        for kwargs in [dict(adapter_size=0), dict(adapter_size=True), dict(dropout=float('nan')),
                       dict(dropout=1.), dict(blocks=[]), dict(blocks=[0, 0]), dict(blocks=[-1]),
                       dict(blocks=[2]), dict(blocks=[False]), dict(blocks=[.5]), dict(freeze_base=1)]:
            model = encoder()
            before = dict(model.named_parameters())
            flags = {n: p.requires_grad for n, p in before.items()}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                add_cbramod_adapters(model, **kwargs)
            self.assertEqual(set(before), set(dict(model.named_parameters())))
            self.assertEqual(flags, {n: p.requires_grad for n, p in model.named_parameters()})
        model = encoder()
        add_cbramod_adapters(model, blocks=[0])
        with self.assertRaises(ValueError):
            add_cbramod_adapters(model)
        self.assertFalse(hasattr(model.encoder.layers[1], 'adapter'))
        for kind in (CBraModClassifier, CBraModRegression, CBraModEmbedding):
            with self.assertRaises(ValueError):
                add_cbramod_adapters(kind(encoder(), 3, 128, num_patches=4, freeze_encoder=True))
        with self.assertRaises(ValueError):
            add_cbramod_adapters(nn.Identity())
        model = encoder()
        model.encoder.layers[0] = nn.Identity()
        with self.assertRaises(ValueError):
            add_cbramod_adapters(model)


if __name__ == '__main__':
    unittest.main()
