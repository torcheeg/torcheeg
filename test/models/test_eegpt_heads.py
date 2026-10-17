import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from torcheeg.models import EEGPT, EEGPTClassifier, EEGPTRegression, EEGPTEmbedding


class TestEEGPTHeads(unittest.TestCase):
    def config(self, **kwargs):
        parameters = dict(num_electrodes=3, channel_names=['C3', 'CZ', 'C4'], chunk_size=128,
                          patch_size=32, hid_channels=16, embed_num=2, depth=2, num_heads=4)
        parameters.update(kwargs)
        return parameters

    def encoder(self, **kwargs):
        return EEGPT(**self.config(**kwargs))

    def test_classification_regression_and_embedding_shapes(self):
        eeg = torch.randn(2, 3, 128)
        for model, shape in [(EEGPTClassifier(self.encoder(), num_classes=3), (2, 3)),
                             (EEGPTRegression(self.encoder()), (2, 1)),
                             (EEGPTRegression(self.encoder(), num_outputs=2), (2, 2)),
                             (EEGPTEmbedding(self.encoder()), (2, 128)),
                             (EEGPTEmbedding(self.encoder(), flatten=False), (2, 4, 2, 16))]:
            with self.subTest(model=type(model), shape=shape):
                self.assertEqual(model(eeg).shape, shape)
                self.assertEqual(model.feature_dim, 128)
                self.assertEqual(model.feature(eeg).shape, (2, 4, 2, 16))
                self.assertEqual(model.feature(eeg, flatten=True).shape, (2, 128))

    def test_feature_and_embedding_match_encoder_without_hidden_projection(self):
        encoder = self.encoder().double().eval()
        model = EEGPTEmbedding(encoder).eval()
        self.assertIs(model.encoder, encoder)
        eeg = torch.randn(2, 3, 128, dtype=torch.float64, requires_grad=True)
        expected = encoder(eeg)
        torch.testing.assert_close(model(eeg), expected.flatten(start_dim=1))
        torch.testing.assert_close(model.feature(eeg), expected)
        model(eeg).square().mean().backward()
        self.assertTrue(torch.isfinite(eeg.grad).all())
        self.assertEqual(set(model.state_dict()), {'encoder.' + name for name in encoder.state_dict()})

    def test_task_heads_match_independent_linear_arithmetic_and_gradients(self):
        for cls in [EEGPTClassifier, EEGPTRegression]:
            with self.subTest(cls=cls):
                model = cls(self.encoder().double(), dropout=0, max_norm=0.5).eval()
                head = model.task_head[1]
                with torch.no_grad():
                    head.weight.copy_(torch.linspace(-3, 2, head.weight.numel()).reshape(head.weight.shape))
                    head.bias.copy_(torch.linspace(-0.2, 0.3, head.bias.numel()))
                reference = copy.deepcopy(model.encoder)
                eeg = torch.randn(2, 3, 128, dtype=torch.float64, requires_grad=True)
                other = eeg.detach().clone().requires_grad_(True)
                weight = torch.renorm(head.weight.detach(), p=2, dim=0, maxnorm=0.5).requires_grad_(True)
                bias = head.bias.detach().clone().requires_grad_(True)
                expected = F.linear(reference(other).flatten(start_dim=1), weight, bias)
                actual = model(eeg)
                torch.testing.assert_close(actual, expected)
                actual.square().mean().backward()
                expected.square().mean().backward()
                torch.testing.assert_close(eeg.grad, other.grad)
                torch.testing.assert_close(head.weight.grad, weight.grad)
                torch.testing.assert_close(head.bias.grad, bias.grad)
                for left, right in zip(model.encoder.parameters(), reference.parameters()):
                    torch.testing.assert_close(left.grad, right.grad)

    def test_freezing_keeps_encoder_eval_while_head_trains(self):
        encoder = self.encoder(dropout=0.8, attn_dropout=0.8, drop_path=0.8)
        model = EEGPTClassifier(encoder, freeze_encoder=True, dropout=0.8)
        snapshot = {name: value.clone() for name, value in encoder.state_dict().items()}
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertTrue(model.task_head.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
        eeg = torch.randn(2, 3, 128, requires_grad=True)
        torch.testing.assert_close(model.feature(eeg), model.feature(eeg))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        head_before = model.task_head[1].bias.detach().clone()
        F.cross_entropy(model(eeg), torch.tensor([0, 1])).backward()
        self.assertTrue(torch.isfinite(eeg.grad).all())
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(head_before, model.task_head[1].bias))
        for name, value in encoder.state_dict().items():
            torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)
        model.eval().train()
        self.assertFalse(model.encoder.training)

    def test_existing_parameter_flags_are_preserved_without_requested_freeze(self):
        encoder = self.encoder()
        encoder.summary_token.requires_grad_(False)
        expected = [parameter.requires_grad for parameter in encoder.parameters()]
        model = EEGPTRegression(encoder)
        model.train()
        self.assertTrue(encoder.training)
        self.assertEqual(expected, [parameter.requires_grad for parameter in encoder.parameters()])

    def test_constraints_preserve_owners_and_accumulated_backward(self):
        model = EEGPTClassifier(self.encoder(), dropout=0)
        head = model.task_head[1]
        parameter = head.weight
        pointer = parameter.data_ptr()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with torch.no_grad():
            parameter.fill_(20)
        eeg = torch.randn(2, 3, 128, requires_grad=True)
        loss = (model(eeg) + model(eeg * 0.8)).square().mean()
        loss.backward()
        self.assertIs(parameter, head.weight)
        self.assertEqual(parameter.data_ptr(), pointer)
        self.assertTrue(torch.all(parameter.detach().norm(dim=1) <= 1.00001))
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertTrue(torch.isfinite(eeg.grad).all())
        optimizer.step()
        self.assertIs(parameter, head.weight)

    def test_disabled_constraints_and_frozen_projection(self):
        eeg = torch.randn(2, 3, 128, requires_grad=True)
        model = EEGPTRegression(self.encoder(), max_norm=None, dropout=0).eval()
        head = model.task_head[1]
        with torch.no_grad():
            head.weight.fill_(20)
        before = head.weight.clone()
        torch.testing.assert_close(model(eeg), F.linear(model.feature(eeg, flatten=True), head.weight, head.bias))
        torch.testing.assert_close(head.weight, before)
        constrained = EEGPTRegression(self.encoder(), max_norm=0.5, dropout=0).eval().requires_grad_(False)
        with torch.no_grad():
            constrained.task_head[1].weight.fill_(20)
        frozen = constrained.task_head[1].weight.clone()
        result = constrained(eeg)
        expected_weight = torch.renorm(frozen, p=2, dim=0, maxnorm=0.5)
        expected = F.linear(constrained.feature(eeg, flatten=True), expected_weight, constrained.task_head[1].bias)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(constrained.task_head[1].weight, frozen, rtol=0, atol=0)
        result.sum().backward()
        self.assertTrue(torch.isfinite(eeg.grad).all())

    def test_checkpoint_encoder_factory_and_full_task_state(self):
        encoder = self.encoder().eval()
        eeg = torch.randn(2, 3, 128)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'encoder.pt'
            torch.save(encoder.state_dict(), path)
            for cls in [EEGPTClassifier, EEGPTRegression, EEGPTEmbedding]:
                with self.subTest(cls=cls):
                    model = cls.from_pretrained(path, encoder_kwargs=self.config(), prefix='', freeze_encoder=True).eval()
                    torch.testing.assert_close(model.feature(eeg), encoder(eeg))
                    self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))
                    restored = cls(self.encoder()).eval()
                    restored.load_state_dict(model.state_dict(), strict=True)
                    torch.testing.assert_close(restored(eeg), model(eeg))
            with self.assertRaises(ValueError):
                EEGPTClassifier.from_pretrained(path, encoder_kwargs=[], prefix='')

    def test_double_dtype_and_registered_parameters(self):
        model = EEGPTRegression(self.encoder().double(), dropout=0)
        self.assertTrue(all(parameter.dtype == torch.float64 for parameter in model.parameters()))
        eeg = torch.randn(2, 3, 128, dtype=torch.float64)
        self.assertEqual(model(eeg).dtype, torch.float64)
        model.float()
        self.assertEqual(model(eeg.float()).dtype, torch.float32)
        self.assertEqual(model.encoder.channel_ids.dtype, torch.long)
        self.assertEqual(len(list(model.parameters())), len(list(model.encoder.parameters())) + 2)

    def test_optimization_updates_encoder_and_head_for_both_tasks(self):
        for cls in [EEGPTClassifier, EEGPTRegression]:
            with self.subTest(cls=cls):
                model = cls(self.encoder(), dropout=0)
                before = model.encoder.patch_embed.proj.weight.detach().clone()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                output = model(torch.randn(2, 3, 128))
                loss = F.cross_entropy(output, torch.tensor([0, 1])) if cls is EEGPTClassifier else (output[:, 0] - torch.tensor([1., -2.])).square().mean()
                loss.backward()
                self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
                optimizer.step()
                self.assertFalse(torch.equal(before, model.encoder.patch_embed.proj.weight))

    def test_strict_export_roundtrip_for_each_task(self):
        eeg = torch.randn(2, 3, 128)
        for cls in [EEGPTClassifier, EEGPTRegression, EEGPTEmbedding]:
            with self.subTest(cls=cls), tempfile.TemporaryDirectory() as directory:
                model = cls(self.encoder()).eval().requires_grad_(False)
                exported = torch.export.export(model, (eeg,), strict=True)
                path = Path(directory) / 'task.pt2'
                torch.export.save(exported, path)
                restored = torch.export.load(path).module()
                torch.testing.assert_close(restored(eeg), model(eeg))

    def test_invalid_settings_do_not_freeze_supplied_encoder(self):
        for cls, kwargs in [(EEGPTClassifier, dict(num_classes=0)), (EEGPTRegression, dict(num_outputs=True)),
                            (EEGPTClassifier, dict(dropout=1)), (EEGPTRegression, dict(max_norm=-1)),
                            (EEGPTRegression, dict(max_norm=float('nan'))), (EEGPTEmbedding, dict(flatten=1))]:
            encoder = self.encoder()
            with self.subTest(cls=cls, kwargs=kwargs), self.assertRaises(ValueError):
                cls(encoder, freeze_encoder=True, **kwargs)
            self.assertTrue(all(parameter.requires_grad for parameter in encoder.parameters()))
        for encoder in [nn.Linear(3, 2), self.encoder(return_attention_layer=1), self.encoder(channel_names=None)]:
            with self.subTest(encoder=type(encoder)), self.assertRaises(ValueError):
                EEGPTClassifier(encoder)
        with self.assertRaises(ValueError):
            EEGPTClassifier(self.encoder(), freeze_encoder=1)
        with self.assertRaises(ValueError):
            EEGPTEmbedding(self.encoder()).feature(torch.randn(2, 3, 128), flatten=1)


if __name__ == '__main__':
    unittest.main()
