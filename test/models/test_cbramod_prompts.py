import copy
import itertools
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from torcheeg.models import (CBraMod, CBraModClassifier, CBraModRegression, CBraModEmbedding,
                             add_cbramod_adapters, add_cbramod_prompts)
from torcheeg.models.transformer import add_cbramod_prompts as TransformerImport


CONFIG = dict(patch_size=25, hid_channels=16, mlp_channels=32, depth=2,
              num_heads=4, dropout=0., position_kernel=(3, 3))


def encoder():
    return CBraMod(**CONFIG)


def task(kind=CBraModClassifier, **kwargs):
    return kind(encoder(), 3, 128, num_patches=4, **kwargs)


class TestCBraModPrompts(unittest.TestCase):
    def test_public_import_base_ownership_and_native_patch_count(self):
        self.assertIs(add_cbramod_prompts, TransformerImport)
        model = encoder().double().eval()
        original = dict(model.named_parameters())
        layers = model.encoder.layers
        self.assertIs(add_cbramod_prompts(model, num_prompts=2, num_electrodes=3), model)
        self.assertIs(model.encoder.layers, layers)
        self.assertEqual(model.encoder.prompt_tokens.shape, (1, 3, 2, 16))
        self.assertEqual(model.encoder.prompt_tokens.numel(), 3 * 2 * 16)
        self.assertEqual(model.encoder.prompt_tokens.dtype, torch.float64)
        after = dict(model.named_parameters())
        self.assertEqual(set(after) - set(original), {'encoder.prompt_tokens'})
        for name, parameter in original.items():
            self.assertIs(after[name], parameter)
            self.assertFalse(parameter.requires_grad)
        self.assertTrue(model.encoder.prompt_tokens.requires_grad)
        self.assertTrue(all(not m.training for m in model.modules()))
        for patches in (2, 4, 6):
            x = torch.randn(2, 3, patches, 25, dtype=torch.float64)
            self.assertEqual(model(x).shape, (2, 3, patches, 200))
            self.assertEqual(model.forward_features(x).shape, (2, 3, patches, 16))

    def test_manual_input_token_operation_matches_output_and_gradients(self):
        model = encoder().double().eval()
        native_encoder = copy.deepcopy(model.encoder).requires_grad_(False)
        add_cbramod_prompts(model, num_prompts=2, num_electrodes=3)
        x = torch.randn(2, 3, 4, 25, dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        actual = model.forward_features(x)
        embedded = model.patch_embedding(y, mask=None)
        prompts = model.encoder.prompt_tokens.expand(2, -1, -1, -1)
        expected = native_encoder(torch.cat((prompts, embedded), dim=2))[:, :, 2:, :]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        coefficient = torch.linspace(-1, 1, actual.numel(), dtype=torch.float64).reshape(actual.shape)
        first = torch.autograd.grad((actual * coefficient).sum(), [x, model.encoder.prompt_tokens])
        second = torch.autograd.grad((expected * coefficient).sum(), [y, model.encoder.prompt_tokens])
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-12)
            self.assertTrue(torch.isfinite(left).all())
            self.assertGreater(left.norm().item(), 0)

    def test_task_geometry_feature_contracts_and_modes(self):
        for kind in (CBraModClassifier, CBraModRegression, CBraModEmbedding):
            with self.subTest(kind=kind.__name__):
                model = task(kind).eval()
                add_cbramod_prompts(model, num_prompts=2)
                self.assertEqual(model.backbone.encoder.num_electrodes, 3)
                x = torch.randn(2, 3, 128)
                features = model.feature(x)
                self.assertEqual(features.shape, (2, 3, 4, 16))
                expected = features.flatten(1) if kind == CBraModEmbedding else model.task_head(features.flatten(1))
                torch.testing.assert_close(model(x), expected)
                model.train()
                self.assertTrue(all(module.training for module in model.modules()))
                model.eval()
                self.assertTrue(all(not module.training for module in model.modules()))

    def test_prompt_insertion_changes_attention_without_changing_output_geometry(self):
        model = encoder().eval()
        x = torch.randn(2, 3, 4, 25)
        before = model.forward_features(x)
        add_cbramod_prompts(model, num_prompts=2, num_electrodes=3)
        with torch.no_grad():
            model.encoder.prompt_tokens.zero_()
        actual = model.forward_features(x)
        self.assertEqual(actual.shape, before.shape)
        self.assertGreater((actual - before).abs().max().item(), 1e-5)

    def test_deepcopy_has_independent_prompt_owners_and_no_captured_original(self):
        model = task(dropout=0.).double().eval()
        add_cbramod_prompts(model, num_prompts=2)
        copied = copy.deepcopy(model)
        self.assertIsNone(getattr(copied.forward, '__closure__', None))
        self.assertNotEqual(model.backbone.encoder.prompt_tokens.data_ptr(), copied.backbone.encoder.prompt_tokens.data_ptr())
        x = torch.randn(2, 3, 128, dtype=torch.float64)
        expected = model(x)
        torch.testing.assert_close(copied(x), expected, rtol=0, atol=0)
        with torch.no_grad():
            copied.backbone.encoder.prompt_tokens[..., 0].add_(2.)
        self.assertGreater((copied(x) - expected).abs().max().item(), 1e-8)
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)

    def test_training_updates_prompts_and_head_while_base_tensors_remain_fixed(self):
        model = task(dropout=0.)
        original = {n: v.clone() for n, v in model.backbone.state_dict().items()}
        add_cbramod_prompts(model, num_prompts=2)
        prompts = model.backbone.encoder.prompt_tokens.detach().clone()
        head = model.task_head.output.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        x = torch.randn(2, 3, 128, requires_grad=True)
        for step in range(2):
            optimizer.zero_grad()
            F.cross_entropy(model(x), torch.tensor([0, 1])).backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.norm().item(), 0)
            self.assertGreater(model.backbone.encoder.prompt_tokens.grad.norm().item(), 0)
            optimizer.step()
        self.assertFalse(torch.equal(prompts, model.backbone.encoder.prompt_tokens))
        self.assertFalse(torch.equal(head, model.task_head.output.weight))
        for name, value in original.items():
            torch.testing.assert_close(model.backbone.state_dict()[name], value, rtol=0, atol=0)

    def test_adapter_composition_both_orders_trainable_keys_and_state_restore(self):
        for order in itertools.permutations(('adapter', 'prompt')):
            with self.subTest(order=order):
                model = task(dropout=0.).double().eval()

                def install(candidate):
                    for method in order:
                        if method == 'adapter':
                            add_cbramod_adapters(candidate, adapter_size=4, dropout=0.)
                        else:
                            add_cbramod_prompts(candidate, num_prompts=2)
                install(model)
                self.assertTrue(all(p.requires_grad == ('.adapter.' in n or n.endswith('prompt_tokens'))
                    for n, p in model.backbone.named_parameters()))
                x = torch.randn(2, 3, 128, dtype=torch.float64)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'augmented.pt'
                    torch.save(model.state_dict(), path)
                    restored = task(dropout=0.).double().eval()
                    install(restored)
                    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
                    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
                model(x).square().mean().backward()
                self.assertGreater(model.backbone.encoder.prompt_tokens.grad.norm().item(), 0)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_existing_flags_preserved_without_freeze_and_head_flags_untouched(self):
        model = task()
        model.backbone.encoder.layers[0].linear1.weight.requires_grad_(False)
        model.task_head.output.bias.requires_grad_(False)
        original = {n: p.requires_grad for n, p in model.named_parameters()}
        add_cbramod_prompts(model, num_prompts=2, freeze_base=False)
        for name, parameter in model.named_parameters():
            if name in original:
                self.assertEqual(parameter.requires_grad, original[name])
        self.assertTrue(model.backbone.encoder.prompt_tokens.requires_grad)

    def test_native_mask_applies_to_signal_patches_and_prompts_still_train(self):
        model = encoder().eval()
        add_cbramod_prompts(model, num_prompts=2, num_electrodes=3)
        x = torch.randn(2, 3, 4, 25, requires_grad=True)
        mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        mask[:, :, 0] = True
        features = model.forward_features(x, mask=mask)
        (features * torch.linspace(-1, 1, features.numel()).reshape(features.shape)).sum().backward()
        torch.testing.assert_close(x.grad[:, :, 0], torch.zeros_like(x.grad[:, :, 0]), rtol=0, atol=0)
        self.assertGreater(x.grad[:, :, 1:].norm().item(), 0)
        self.assertGreater(model.encoder.prompt_tokens.grad.norm().item(), 0)

    def test_invalid_configuration_duplicates_and_custom_encoder_fail_before_mutation(self):
        for config in [dict(num_prompts=0), dict(num_prompts=True), dict(num_electrodes=0),
                       dict(num_electrodes=4), dict(num_electrodes=False), dict(freeze_base=1)]:
            model = task()
            before = {n: p.requires_grad for n, p in model.named_parameters()}
            with self.subTest(config=config), self.assertRaises(ValueError):
                add_cbramod_prompts(model, **config)
            self.assertEqual(before, {n: p.requires_grad for n, p in model.named_parameters()})
        with self.assertRaises(ValueError):
            add_cbramod_prompts(encoder())
        model = encoder()
        model.encoder = nn.Identity()
        with self.assertRaises(ValueError):
            add_cbramod_prompts(model, num_electrodes=3)
        with self.assertRaises(ValueError):
            add_cbramod_prompts(task(freeze_encoder=True))
        model = task()
        add_cbramod_prompts(model, num_prompts=2)
        before = {n: p.requires_grad for n, p in model.named_parameters()}
        with self.assertRaises(ValueError):
            add_cbramod_prompts(model)
        self.assertEqual(before, {n: p.requires_grad for n, p in model.named_parameters()})
        bare = encoder()
        add_cbramod_prompts(bare, num_electrodes=3)
        with self.assertRaises(ValueError):
            bare(torch.randn(2, 4, 4, 25))


if __name__ == '__main__':
    unittest.main()
