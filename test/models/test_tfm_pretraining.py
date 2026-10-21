import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from torcheeg.models import (
    TFMTokenizer,
    TFMEncoder,
    TFMTokenizerPretrainer,
    TFMMaskedTokenPretrainer,
)
from test.models.test_tfm_tokenizer import controlled_dropout


class TestTFMPretraining(unittest.TestCase):
    def tokenizer(self):
        return TFMTokenizer(
            hid_channels=16,
            num_tokens=23,
            encoder_depth=1,
            decoder_depth=1,
            reconstruction_head='mlp',
        )

    def encoder(self):
        return TFMEncoder(hid_channels=16, num_tokens=23, num_layers=1, num_classes=23)

    def test_independent_native_fixture_loss_outputs_every_gradient_and_ema(self):
        fixture = json.loads(
            (
                Path(__file__).parent / 'fixtures/tfm_pretraining_reference.json'
            ).read_text()
        )

        def tensor(item):
            return torch.tensor(
                item['values'], dtype=getattr(torch, item['dtype'])
            ).reshape(item['shape'])

        for case in fixture['cases']:
            with self.subTest(objective=case['objective'], training=case['training']):
                tokenizer = TFMTokenizer(**fixture['tokenizer_config']).double()
                tokenizer.load_state_dict(
                    {n: tensor(t) for n, t in fixture['tokenizer_state'].items()},
                    strict=True,
                )
                if case['objective'] == 'mtp':
                    encoder = TFMEncoder(**fixture['encoder_config']).double()
                    encoder.load_state_dict(
                        {n: tensor(t) for n, t in fixture['encoder_state'].items()},
                        strict=True,
                    )
                    model = TFMMaskedTokenPretrainer(tokenizer, encoder)
                else:
                    model = TFMTokenizerPretrainer(tokenizer)
                model.train(case['training'])
                eeg = tensor(fixture['input']).requires_grad_()
                with controlled_dropout():
                    output = model(eeg, mask=tensor(case['mask']))
                for name in (
                    'loss',
                    'reconstruction_loss',
                    'quantization_loss',
                    'reconstruction',
                    'complementary_reconstruction',
                    'prediction',
                    'complementary_prediction',
                ):
                    if name in case:
                        torch.testing.assert_close(
                            output[name],
                            tensor(case[name]),
                            rtol=1e-8,
                            atol=1e-10,
                            msg=name,
                        )
                output['loss'].backward()
                if case['input_gradient'] is None:
                    self.assertIsNone(eeg.grad)
                else:
                    torch.testing.assert_close(
                        eeg.grad, tensor(case['input_gradient']), rtol=1e-7, atol=1e-9
                    )
                self.assertEqual(
                    set(dict(model.named_parameters())),
                    set(case['parameter_gradients']),
                )
                for name, parameter in model.named_parameters():
                    expected = case['parameter_gradients'][name]
                    if expected is None:
                        self.assertIsNone(parameter.grad, name)
                    else:
                        torch.testing.assert_close(
                            parameter.grad,
                            tensor(expected),
                            rtol=1e-7,
                            atol=1e-9,
                            msg=name,
                        )
                for name, value in tokenizer.state_dict().items():
                    torch.testing.assert_close(
                        value,
                        tensor(case['tokenizer_state_after'][name]),
                        rtol=1e-9,
                        atol=1e-11,
                        msg=name,
                    )

    def test_spectral_mask_native_shared_groups_and_complement(self):
        spectrum = torch.ones(4, 100, 7)
        before = torch.random.get_rng_state()
        mask = TFMTokenizerPretrainer.mask_spectrum(
            spectrum, torch.Generator().manual_seed(13)
        )
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(mask[0], mask[3]))
        self.assertTrue(
            torch.equal(
                mask.reshape(4, 20, 5, 7),
                mask.reshape(4, 20, 5, 7)[:, :, :1].expand(-1, -1, 5, -1),
            )
        )
        self.assertEqual(mask[0].sum().item(), 10 * 5 * 4)
        self.assertTrue(torch.equal(spectrum * mask + spectrum * ~mask, spectrum))
        for frames in (1, 2, 3, 7):
            generated = TFMTokenizerPretrainer.mask_spectrum(torch.ones(1, 100, frames))
            self.assertEqual(generated.sum().item(), 50 * (frames - frames // 2))

    def test_native_quantization_cancellation_and_explicit_ema_gradient(self):
        tokenizer = self.tokenizer().eval()
        native = TFMTokenizerPretrainer(tokenizer, quantization_mode='native')
        corrected = TFMTokenizerPretrainer(
            copy.deepcopy(tokenizer), quantization_mode='ema'
        )
        eeg = torch.randn(2, 2, 400)
        mask = TFMTokenizerPretrainer.mask_spectrum(torch.ones(4, 100, 3))
        left = native(eeg, mask=mask)
        right = corrected(eeg, mask=mask)
        self.assertAlmostEqual(
            left['quantization_loss'].item(),
            2 * right['quantization_loss'].item(),
            places=6,
        )
        left['quantization_loss'].backward()
        right['quantization_loss'].backward()
        self.assertEqual(
            sum(
                p.grad.abs().sum().item()
                for p in tokenizer.parameters()
                if p.grad is not None
            ),
            0.0,
        )
        self.assertGreater(
            sum(
                p.grad.abs().sum().item()
                for p in corrected.parameters()
                if p.grad is not None
            ),
            0.0,
        )
        self.assertFalse(tokenizer.quantizer.embedding.weight.requires_grad)

    def test_symmetric_loss_mask_partition_mean_and_frozen_teacher(self):
        model = TFMMaskedTokenPretrainer(self.tokenizer(), self.encoder()).train()
        before = copy.deepcopy(model.tokenizer.state_dict())
        eeg = torch.randn(2, 2, 400, requires_grad=True)
        mask = torch.zeros(2, 2, 3, dtype=torch.bool)
        mask[0, 0, 0] = True
        output = model(eeg, mask=mask)
        ids = output['ids']
        first = torch.nn.functional.cross_entropy(
            output['prediction'].reshape(-1, 23)[~mask.flatten()], ids[~mask]
        )
        second = torch.nn.functional.cross_entropy(
            output['complementary_prediction'].reshape(-1, 23)[mask.flatten()],
            ids[mask],
        )
        torch.testing.assert_close(output['loss'], first + second, rtol=0, atol=0)
        self.assertFalse(torch.equal(output['loss'], (11 * first + second) / 12))
        output['loss'].backward()
        self.assertIsNone(eeg.grad)
        self.assertFalse(model.tokenizer.training)
        self.assertTrue(all(p.grad is None for p in model.tokenizer.parameters()))
        self.assertGreater(
            model.encoder.eeg_token_embedding.weight.grad[23].abs().sum().item(), 0.0
        )
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.encoder.parameters()
                if p.requires_grad
            )
        )
        for name, value in before.items():
            torch.testing.assert_close(
                value, model.tokenizer.state_dict()[name], rtol=0, atol=0
            )

    def test_full_model_optimizer_mask_and_dropout_rng_resume(self):
        eeg = torch.randn(2, 2, 400)
        for objective, mode in (
            ('tokenizer', 'native'),
            ('tokenizer', 'ema'),
            ('mtp', None),
        ):
            with self.subTest(objective=objective, quantization_mode=mode):
                model = (
                    TFMTokenizerPretrainer(self.tokenizer(), quantization_mode=mode)
                    if objective == 'tokenizer'
                    else TFMMaskedTokenPretrainer(self.tokenizer(), self.encoder())
                )
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad], lr=1e-4
                )
                generator = torch.Generator().manual_seed(19)
                model(eeg, generator=generator)['loss'].backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'resume.pt'
                    torch.save(
                        dict(
                            model=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            mask_rng=generator.get_state(),
                            rng=torch.random.get_rng_state(),
                        ),
                        path,
                    )
                    payload = torch.load(path, weights_only=True)
                    other = (
                        TFMTokenizerPretrainer(self.tokenizer(), quantization_mode=mode)
                        if objective == 'tokenizer'
                        else TFMMaskedTokenPretrainer(self.tokenizer(), self.encoder())
                    )
                    other.load_state_dict(payload['model'], strict=True)
                    second = torch.optim.AdamW(
                        [p for p in other.parameters() if p.requires_grad], lr=1e-4
                    )
                    second.load_state_dict(payload['optimizer'])
                    rng = torch.Generator().set_state(payload['mask_rng'])
                    for current, opt, gen in (
                        (model, optimizer, generator),
                        (other, second, rng),
                    ):
                        torch.random.set_rng_state(payload['rng'])
                        current(eeg, generator=gen)['loss'].backward()
                        opt.step()
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(
                            value, other.state_dict()[name], rtol=0, atol=0, msg=name
                        )
                    torch.testing.assert_close(
                        generator.get_state(), rng.get_state(), rtol=0, atol=0
                    )

    def test_aot_eager_complete_dual_view_training_backward_and_state(self):
        for objective in ('tokenizer', 'mtp'):
            with self.subTest(objective=objective):
                model = (
                    TFMTokenizerPretrainer(self.tokenizer())
                    if objective == 'tokenizer'
                    else TFMMaskedTokenPretrainer(self.tokenizer(), self.encoder())
                )
                reference = copy.deepcopy(model).train()
                model.train()
                x = torch.randn(2, 2, 400, requires_grad=True)
                y = x.detach().clone().requires_grad_()
                if objective == 'tokenizer':
                    mask = TFMTokenizerPretrainer.mask_spectrum(
                        torch.ones(4, 100, 3), torch.Generator().manual_seed(13)
                    )
                else:
                    mask = torch.arange(12).reshape(2, 2, 3) % 2 == 0
                torch.manual_seed(17)
                expected = reference(x, mask=mask)
                compiled = torch.compile(model, backend='aot_eager', fullgraph=False)
                torch.manual_seed(17)
                actual = compiled(y, mask=mask)
                for name, value in expected.items():
                    torch.testing.assert_close(
                        actual[name], value, rtol=1e-4, atol=1e-5, msg=name
                    )
                expected['loss'].backward()
                actual['loss'].backward()
                if x.grad is None:
                    self.assertIsNone(y.grad)
                else:
                    torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=1e-5)
                parameters = dict(model.named_parameters())
                for name, parameter in reference.named_parameters():
                    other = parameters[name]
                    if parameter.grad is None:
                        self.assertIsNone(other.grad, name)
                    else:
                        torch.testing.assert_close(
                            parameter.grad, other.grad, rtol=1e-4, atol=1e-5, msg=name
                        )
                for name, value in reference.state_dict().items():
                    torch.testing.assert_close(
                        value, model.state_dict()[name], rtol=1e-5, atol=1e-6, msg=name
                    )

    def test_invalid_input_objective_vocabulary_and_degenerate_partitions(self):
        with self.assertRaises(ValueError):
            TFMTokenizerPretrainer(self.tokenizer(), quantization_mode='unknown')
        with self.assertRaisesRegex(ValueError, 'vocabulary'):
            TFMMaskedTokenPretrainer(
                self.tokenizer(),
                TFMEncoder(hid_channels=16, num_tokens=23, num_layers=1),
            )
        tokenizer = TFMTokenizerPretrainer(self.tokenizer())
        mtp = TFMMaskedTokenPretrainer(self.tokenizer(), self.encoder())
        for model in (tokenizer, mtp):
            for eeg in (
                None,
                torch.randn(2, 2, 199),
                torch.ones(2, 2, 400, dtype=torch.long),
                torch.randn(0, 2, 400),
            ):
                with self.assertRaises(ValueError):
                    model(eeg)
        eeg = torch.randn(2, 2, 400)
        for mask in (
            torch.ones(2, 2, 3, dtype=torch.bool),
            torch.zeros(2, 2, 3, dtype=torch.bool),
        ):
            with self.assertRaisesRegex(ValueError, 'nonempty'):
                mtp(eeg, mask=mask)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            mtp(torch.randn(1, 1, 200))
        with self.assertRaisesRegex(ValueError, '1--16'):
            mtp(torch.randn(1, 17, 400))
        for model in (tokenizer, mtp):
            with self.assertRaisesRegex(ValueError, 'Boolean'):
                model(eeg, mask=torch.ones(2, 2, 3))


if __name__ == '__main__':
    unittest.main()
