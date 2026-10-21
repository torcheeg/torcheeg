import copy
import json
from argparse import Namespace
from pathlib import Path
import pickle
import tempfile
import unittest

import torch

from torcheeg.models import STEEGFormerPretrainer


class TestSTEEGFormerPretrainer(unittest.TestCase):
    config = dict(
        patch_size=4,
        hid_channels=8,
        mlp_channels=32,
        depth=1,
        num_heads=2,
        decoder_channels=8,
        decoder_depth=1,
        decoder_num_heads=2,
    )

    def build(self, **kwargs):
        return STEEGFormerPretrainer(**dict(self.config, **kwargs))

    def test_independent_native_outputs_loss_input_and_all_parameter_gradients(self):
        fixture = json.loads(
            (
                Path(__file__).parent
                / 'fixtures'
                / 'steegformer_pretraining_reference.json'
            ).read_text()
        )

        def tensor(item):
            return torch.tensor(item['values'], dtype=torch.float64).reshape(
                item['shape']
            )

        indices = torch.tensor(fixture['channel_indices'])
        for case in fixture['cases']:
            with self.subTest(
                offset=case['offset'],
                norm=case['norm_pix_loss'],
                ratio=case['mask_ratio'],
            ):
                model = (
                    STEEGFormerPretrainer(
                        **fixture['config'],
                        encoder_position_offset=case['offset'],
                        norm_pix_loss=case['norm_pix_loss'],
                    )
                    .double()
                    .eval()
                )
                model.load_state_dict(
                    {n: tensor(t) for n, t in fixture['state_dict'].items()},
                    strict=True,
                )
                eeg = tensor(fixture['input']).requires_grad_()
                generator = torch.Generator().manual_seed(fixture['mask_seed'])
                loss, prediction, mask = model(
                    eeg, indices, case['mask_ratio'], generator
                )
                torch.testing.assert_close(
                    mask, tensor(case['mask']), rtol=0, atol=0, check_dtype=False
                )
                torch.testing.assert_close(
                    loss, tensor(case['loss']), rtol=1e-10, atol=1e-11
                )
                torch.testing.assert_close(
                    prediction, tensor(case['prediction']), rtol=1e-10, atol=1e-11
                )
                coefficient = torch.linspace(
                    -0.7, 0.9, prediction.numel(), dtype=prediction.dtype
                ).reshape_as(prediction)
                (loss + (prediction * coefficient).mean()).backward()
                torch.testing.assert_close(
                    eeg.grad, tensor(case['input_gradient']), rtol=1e-9, atol=1e-10
                )
                for name, parameter in model.named_parameters():
                    expected = case['parameter_gradients'][name]
                    if expected is None:
                        self.assertIsNone(parameter.grad, name)
                    else:
                        torch.testing.assert_close(
                            parameter.grad,
                            tensor(expected),
                            rtol=1e-9,
                            atol=1e-10,
                            msg=name,
                        )

    def test_patch_order_inverse_and_masked_only_loss(self):
        model = self.build()
        eeg = torch.arange(24.0).reshape(1, 2, 12)
        patches = model.patchify(eeg)
        torch.testing.assert_close(
            patches[0, :2],
            torch.tensor([[0.0, 1.0, 2.0, 3.0], [12.0, 13.0, 14.0, 15.0]]),
        )
        torch.testing.assert_close(model.unpatchify(patches, 2), eeg, rtol=0, atol=0)
        mask = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
        prediction = patches.clone().requires_grad_()
        with torch.no_grad():
            prediction.add_(torch.where(mask.unsqueeze(-1).bool(), 2.0, 50.0))
        loss = model.forward_loss(eeg, prediction, mask)
        self.assertEqual(loss.item(), 4.0)
        loss.backward()
        self.assertEqual(prediction.grad[mask == 0].count_nonzero().item(), 0)
        self.assertTrue((prediction.grad[mask == 1] > 0).all())
        model.norm_pix_loss = True
        expected = (patches - patches.mean(-1, keepdim=True)) / (
            patches.var(-1, keepdim=True) + 1e-6
        ).sqrt()
        self.assertEqual(model.forward_loss(eeg, expected, mask).item(), 0.0)

    def test_masking_permutation_generator_and_global_rng_isolation(self):
        model = self.build()
        tokens = torch.arange(2 * 10 * 8.0).reshape(2, 10, 8)
        state = torch.random.get_rng_state()
        visible, mask, restore = model.random_masking(
            tokens, 0.6, torch.Generator().manual_seed(42)
        )
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(visible.shape, (2, 4, 8))
        self.assertTrue(torch.equal(mask.sum(1), torch.tensor([6.0, 6.0])))
        padded = torch.cat((visible, torch.zeros(2, 6, 8)), 1)
        restored = torch.gather(padded, 1, restore.unsqueeze(-1).expand(-1, -1, 8))
        torch.testing.assert_close(
            restored[mask == 0], tokens[mask == 0], rtol=0, atol=0
        )
        self.assertFalse(torch.equal(mask[0], mask[1]))
        other = model.random_masking(tokens, 0.6, torch.Generator().manual_seed(42))
        for a, b in zip((visible, mask, restore), other):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_batch_channel_indices_and_both_position_contracts(self):
        model = self.build().eval()
        eeg = torch.randn(2, 2, 12)
        indices = torch.tensor([6, 60])
        outputs = []
        for channels in (indices, indices[None], indices.repeat(2, 1)):
            outputs.append(
                model(eeg, channels, generator=torch.Generator().manual_seed(13))
            )
        for output in outputs[1:]:
            for a, b in zip(output, outputs[0]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        raw = model.forward_all(eeg, indices)
        model.encoder_position_offset = 1
        torch.testing.assert_close(raw, model.forward_all(eeg, indices), rtol=0, atol=0)
        self.assertFalse(
            torch.equal(
                outputs[0][1],
                model(eeg, indices, generator=torch.Generator().manual_seed(13))[1],
            )
        )

    def test_complete_training_optimizer_and_mask_rng_resume(self):
        torch.manual_seed(17)
        model = self.build()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        generator = torch.Generator().manual_seed(19)
        eeg, indices = torch.randn(2, 2, 12), torch.tensor([6, 60])
        before = copy.deepcopy(model.state_dict())
        model(eeg, indices, generator=generator)[0].backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.assertTrue(
            any(
                not torch.equal(before[n], t)
                for n, t in model.state_dict().items()
                if n.startswith('decoder_blocks.')
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'resume.pt'
            torch.save(
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    generator=generator.get_state(),
                    config=self.config,
                    norm_pix_loss=False,
                    encoder_position_offset=0,
                ),
                path,
            )
            payload = torch.load(path, weights_only=True)
            restored = STEEGFormerPretrainer.from_pretrained(path, **payload['config'])
            second_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
            second_optimizer.load_state_dict(payload['optimizer'])
            second_generator = torch.Generator().set_state(payload['generator'])
            for current, opt, rng in (
                (model, optimizer, generator),
                (restored, second_optimizer, second_generator),
            ):
                current(eeg, indices, generator=rng)[0].backward()
                opt.step()
            for name, value in model.state_dict().items():
                torch.testing.assert_close(
                    value, restored.state_dict()[name], rtol=0, atol=0, msg=name
                )
            self.assertTrue(
                torch.equal(generator.get_state(), second_generator.get_state())
            )

    def test_complete_native_variant_geometry(self):
        for variant, count, width, decoder_width in (
            ('small', 160, 512, 384),
            ('base', 256, 768, 512),
            ('large', 400, 1024, 512),
        ):
            with self.subTest(variant=variant), torch.device('meta'):
                model = STEEGFormerPretrainer.for_variant(variant)
                self.assertEqual(len(model.state_dict()), count)
                self.assertEqual(
                    model.decoder_embed.weight.shape, (decoder_width, width)
                )
                self.assertEqual(model.decoder_pred.weight.shape, (16, decoder_width))
                self.assertEqual(
                    model.dec_channel_emd.channel_transformation.weight.shape,
                    (145, decoder_width),
                )

    def test_loader_full_state_prefix_metadata_and_rejections(self):
        model = self.build()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            for container in (None, 'model', 'state_dict'):
                torch.save(
                    model.state_dict()
                    if container is None
                    else {container: model.state_dict()},
                    path,
                )
                actual = STEEGFormerPretrainer.from_pretrained(path, **self.config)
                for name, tensor in model.state_dict().items():
                    torch.testing.assert_close(
                        tensor, actual.state_dict()[name], rtol=0, atol=0
                    )
            state = {'backbone.' + n: t for n, t in model.state_dict().items()}
            state['head.weight'] = torch.ones(1)
            torch.save(state, path)
            STEEGFormerPretrainer.from_pretrained(
                path, prefix='backbone.', **self.config
            )
            variants = []
            missing = dict(model.state_dict())
            missing.pop('decoder_pred.weight')
            variants.append(missing)
            extra = dict(model.state_dict())
            extra['unrecognized'] = torch.ones(1)
            variants.append(extra)
            for tensor in (
                torch.ones(2),
                torch.ones(4, dtype=torch.long),
                torch.full((4,), float('nan')),
            ):
                invalid = dict(model.state_dict())
                invalid['decoder_pred.bias'] = tensor
                variants.append(invalid)
            variants.extend(
                [{'model': {}, 'state_dict': {}}, {'model': {1: torch.ones(1)}}, []]
            )
            for state in variants:
                torch.save(state, path)
                with self.assertRaises(ValueError):
                    STEEGFormerPretrainer.from_pretrained(path, **self.config)
            torch.save(dict(model=model.state_dict(), args=Namespace(epochs=1)), path)
            with self.assertRaises(pickle.UnpicklingError):
                STEEGFormerPretrainer.from_pretrained(path, **self.config)
            if hasattr(torch.serialization, 'safe_globals'):
                before = torch.serialization.get_safe_globals()
                STEEGFormerPretrainer.from_pretrained(
                    path, allow_namespace=True, **self.config
                )
                self.assertCountEqual(before, torch.serialization.get_safe_globals())
                with torch.serialization.safe_globals([Namespace]):
                    before = torch.serialization.get_safe_globals()
                    STEEGFormerPretrainer.from_pretrained(
                        path, allow_namespace=True, **self.config
                    )
                    self.assertCountEqual(
                        before, torch.serialization.get_safe_globals()
                    )

    def test_aot_eager_complete_masked_training_outputs_and_all_gradients(self):
        model = self.build()
        reference = copy.deepcopy(model)
        eeg = torch.randn(2, 2, 12, requires_grad=True)
        other = eeg.detach().clone().requires_grad_()
        indices = torch.tensor([6, 60])
        torch.manual_seed(18)
        expected = reference(eeg, indices, 0.5)
        compiled = torch.compile(model, backend='aot_eager', fullgraph=True)
        torch.manual_seed(18)
        actual = compiled(other, indices, 0.5)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        actual[0].backward()
        expected[0].backward()
        torch.testing.assert_close(other.grad, eeg.grad, rtol=1e-4, atol=1e-6)
        parameters = dict(model.named_parameters())
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(
                parameters[name].grad, parameter.grad, rtol=1e-4, atol=1e-6, msg=name
            )

    def test_invalid_policy_and_signal_inputs(self):
        for kwargs in (
            {'decoder_channels': 7},
            {'decoder_depth': True},
            {'decoder_num_heads': 0},
            {'norm_pix_loss': 1},
            {'encoder_position_offset': 2},
            {'encoder_position_offset': 0.0},
            {'dropout': 0.1},
            {'qkv_bias': False},
            {'norm_pix_loss': True, 'patch_size': 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build(**kwargs)
        for variant in ('other', None, []):
            with self.assertRaises(ValueError):
                STEEGFormerPretrainer.for_variant(variant)
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            STEEGFormerPretrainer.for_variant('small', num_heads=4)
        model = self.build(max_patches=3)
        for mask in (
            torch.zeros(2, 6),
            torch.full((2, 6), float('nan')),
            torch.full((2, 6), 0.5),
        ):
            with self.assertRaisesRegex(ValueError, 'binary'):
                model.forward_loss(torch.randn(2, 2, 12), torch.randn(2, 6, 4), mask)
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            model.random_masking(torch.randn(2, 3))
        eeg, indices = torch.randn(2, 2, 12), torch.tensor([6, 60])
        for ratio in (0.0, -0.1, 1.1, True, float('nan')):
            with self.assertRaises(ValueError):
                model(eeg, indices, ratio)
        for value in (
            torch.randn(2, 2, 13),
            torch.randn(2, 2, 16),
            torch.ones(2, 2, 12, dtype=torch.long),
            None,
        ):
            with self.assertRaises(ValueError):
                model(value, indices)
        for channels in (
            None,
            [6, 60],
            torch.tensor([6.0]),
            torch.tensor([[6, 60]]).repeat(3, 1),
        ):
            with self.assertRaises(ValueError):
                model(eeg, channels)
        for channels in (torch.tensor([-1, 60]), torch.tensor([6, 145])):
            with self.assertRaises((IndexError, RuntimeError)):
                model(eeg, channels)


if __name__ == '__main__':
    unittest.main()
