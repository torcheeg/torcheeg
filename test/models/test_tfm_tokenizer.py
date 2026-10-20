from contextlib import contextmanager
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from safetensors.torch import save_file

from torcheeg.models import TFMTokenizer, TFMEncoder, TFMClassifier
from torcheeg.models.transformer.tfm_tokenizer import _EMAQuantizer


class SmallTokenizer(TFMTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**dict(dict(num_tokens=23), **kwargs))


class SmallEncoder(TFMEncoder):
    def __init__(self, **kwargs):
        super().__init__(**dict(dict(num_tokens=23), **kwargs))


def _distributed_ema_worker(rank, directory):
    root = Path(directory)
    quantizer = _EMAQuantizer(7, 8, .99, True).train()
    with torch.no_grad():
        quantizer.embedding.weight.zero_()
        quantizer.ema_w.zero_()
    distributed.init_process_group('gloo', init_method=(root/'rendezvous').as_uri(),
        rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        quantizer(torch.full((1, 3, 8), float(rank+1)))
        torch.save(quantizer.state_dict(), root/f'rank-{rank}.pt')
    finally:
        distributed.destroy_process_group()


def reference_state(state):
    result = {}
    for name, value in state.items():
        if name == 'index' or name.endswith('.pe'):
            result[name] = value.clone()
        elif name.endswith('cluster_size'):
            result[name] = torch.ones_like(value)
        elif name.endswith('bias'):
            result[name] = torch.zeros_like(value)
        elif name.endswith('.weight') and value.ndim == 1:
            result[name] = torch.ones_like(value)
        else:
            offset = int(hashlib.sha256(name.encode()).hexdigest()[:6], 16) % 97
            result[name] = (torch.sin(torch.arange(value.numel(), dtype=torch.float64)/17+offset)*.02).reshape_as(value)
    if 'quantizer.embedding.weight' in result:
        result['quantizer.ema_w'] = result['quantizer.embedding.weight'].clone()
    return result


@contextmanager
def controlled_dropout():
    def forward(module, x):
        if not module.training or module.p == 0:
            return x
        keep = torch.arange(x.numel(), device=x.device).reshape_as(x) % 10 >= module.p*10
        return x*keep.to(x.dtype)/(1-module.p)
    with patch.object(torch.nn.Dropout, 'forward', forward):
        yield


class TestTFMTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((Path(__file__).parent/'fixtures/tfm_reference.json').read_text())

    def check_gradients(self, model, expected):
        self.assertEqual(set(dict(model.named_parameters())), set(expected))
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                record = expected[name]
                if record is None:
                    self.assertIsNone(parameter.grad)
                    continue
                gradient = parameter.grad.flatten()
                torch.testing.assert_close(gradient[:8], torch.tensor(record['first'], dtype=torch.float64), rtol=1e-7, atol=1e-8)
                torch.testing.assert_close(gradient.mean(), torch.tensor(record['sum']/gradient.numel(), dtype=torch.float64), rtol=1e-7, atol=1e-8)
                torch.testing.assert_close(gradient.norm(), torch.tensor(record['norm'], dtype=torch.float64), rtol=1e-7, atol=1e-8)

    def test_native_tokenizer_outputs_every_gradient_and_ema(self):
        model = SmallTokenizer().double()
        model.load_state_dict(reference_state(model.state_dict()))
        self.assertEqual(len(model.state_dict()), 189)
        for mode in ['eval', 'train']:
            model.train(mode == 'train'); model.zero_grad(set_to_none=True)
            temporal = torch.sin(torch.arange(1600, dtype=torch.float64)/17).reshape(2, 800).requires_grad_()
            spectrum = torch.sin(torch.arange(1400, dtype=torch.float64)/31).reshape(2, 100, 7).abs().requires_grad_()
            with controlled_dropout():
                outputs = model(spectrum, temporal)
            loss = outputs[0].square().mean() + model.quantization_loss(outputs[3], outputs[2])
            loss.backward()
            record = self.fixture['tokenizer'][mode]
            self.assertAlmostEqual(loss.item(), record['loss'], places=8)
            for index, output in enumerate(outputs):
                torch.testing.assert_close(output, torch.tensor(record['outputs'][index], dtype=output.dtype), rtol=1e-7, atol=1e-8)
            torch.testing.assert_close(temporal.grad, torch.tensor(record['temporal_gradient'], dtype=torch.float64), rtol=1e-7, atol=1e-8)
            torch.testing.assert_close(spectrum.grad, torch.tensor(record['spectrum_gradient'], dtype=torch.float64), rtol=1e-7, atol=1e-8)
            self.check_gradients(model, record['gradients'])
            for actual, key in [(model.quantizer.cluster_size, 'updated_cluster'),
                                (model.quantizer.embedding.weight, 'updated_codebook'), (model.quantizer.ema_w, 'updated_ema')]:
                torch.testing.assert_close(actual, torch.tensor(record[key], dtype=torch.float64), rtol=1e-7, atol=1e-8)

    def test_native_encoder_logits_token_predictions_and_every_gradient(self):
        model = SmallEncoder().double()
        model.load_state_dict(reference_state(model.state_dict()))
        self.assertEqual(len(model.state_dict()), 59)
        ids = torch.arange(42).reshape(2, 3, 7)*7 % 24
        for mode in ['eval', 'train', 'masked_prediction']:
            model.train(mode == 'train'); model.zero_grad(set_to_none=True)
            with controlled_dropout():
                output = model.masked_prediction(ids) if mode == 'masked_prediction' else model(ids)
            output.square().mean().backward()
            record = self.fixture['encoder'][mode]
            torch.testing.assert_close(output, torch.tensor(record['output'], dtype=torch.float64), rtol=1e-7, atol=1e-8)
            self.check_gradients(model, record['gradients'])

    def test_ema_commitment_has_encoder_gradient(self):
        model = SmallTokenizer()
        vectors = torch.tensor([[[.2, -.3]]], requires_grad=True)
        quantized = vectors + (torch.tensor([[[.8, .5]]])-vectors).detach()
        loss = model.quantization_loss(vectors, quantized)
        loss.backward()
        torch.testing.assert_close(vectors.grad, torch.tensor([[[-.6, -.8]]]), rtol=0, atol=1e-7)
        self.assertFalse(model.quantizer.embedding.weight.requires_grad)
        original = torch.tensor([[[.2, -.3]]], requires_grad=True)
        straight = original + (torch.tensor([[[.8, .5]]])-original).detach()
        native_loss = (straight-original.detach()).square().mean() + (straight.detach()-original).square().mean()
        native_loss.backward()
        self.assertEqual(original.grad.count_nonzero().item(), 0)

    def test_single_frame_zero_input_batch_isolation_and_optimizer(self):
        model = SmallTokenizer().eval()
        spectrum = torch.zeros(2, 100, 1, requires_grad=True)
        temporal = torch.zeros(2, 200, requires_grad=True)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        prediction = model(spectrum, temporal)[0]
        self.assertTrue(torch.isfinite(prediction).all())
        prediction[0].sum().backward()
        self.assertEqual(spectrum.grad[1].count_nonzero().item(), 0)
        self.assertEqual(temporal.grad[1].count_nonzero().item(), 0)
        with torch.no_grad():
            torch.testing.assert_close(model(spectrum[:1], temporal[:1])[0], prediction[:1], rtol=1e-5, atol=1e-6)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        model.train(); optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad(); output = model(torch.randn(2, 100, 3), torch.randn(2, 400))
        loss = output[0].square().mean()+model.quantization_loss(output[3], output[2]); loss.backward(); optimizer.step()
        self.assertFalse(torch.equal(model.decoder.weight, before['decoder.weight']))
        self.assertTrue((model.quantizer.cluster_size > 0).any())

    def test_encoder_channel_rows_and_reserved_mask(self):
        model = SmallEncoder().eval()
        ids = torch.arange(21).reshape(1, 3, 7) % 24
        ids[0, 0, 0] = 23
        expected = model(ids)
        torch.testing.assert_close(model(ids, [0, 1, 2]), expected, rtol=0, atol=0)
        torch.testing.assert_close(model(ids[:, [2, 0, 1]], [2, 0, 1]), expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(model(ids[:, :1, :1]).shape, (1, 1))
        self.assertEqual(model.masked_prediction(ids).shape, (1, 21, 1))
        trained_mtp = SmallEncoder(num_classes=23)
        self.assertEqual(trained_mtp.masked_prediction(ids).shape, (1, 21, 23))
        clean = torch.arange(21).reshape(1, 3, 7) % 23
        mask = torch.zeros_like(clean, dtype=torch.bool); mask[:, 0, 0] = True
        masked = clean.clone(); masked[mask] = 23
        predictions = trained_mtp.masked_prediction(masked).reshape(1, 3, 7, 23)
        torch.nn.functional.cross_entropy(predictions[mask], clean[mask]).backward()
        self.assertGreater(trained_mtp.eeg_token_embedding.weight.grad[23].abs().sum().item(), 0)
        self.assertGreater(trained_mtp.classification_head.weight.grad.abs().sum().item(), 0)

    @unittest.skipUnless(distributed.is_available() and distributed.is_gloo_available(), 'CPU Gloo is unavailable.')
    def test_distributed_ema_matches_global_batch(self):
        expected = _EMAQuantizer(7, 8, .99, True).train()
        with torch.no_grad():
            expected.embedding.weight.zero_()
            expected.ema_w.zero_()
        expected(torch.cat([torch.ones(1, 3, 8), torch.full((1, 3, 8), 2.)]))
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.spawn(_distributed_ema_worker, args=(directory,), nprocs=2, join=False)
            deadline = time.monotonic()+60
            while not context.join(timeout=1):
                if time.monotonic() >= deadline:
                    for process in context.processes: process.terminate()
                    self.fail('Distributed EMA exceeded60seconds.')
            for rank in range(2):
                state = torch.load(Path(directory)/f'rank-{rank}.pt', weights_only=True)
                for name, value in expected.state_dict().items():
                    torch.testing.assert_close(state[name], value, rtol=0, atol=0)

    def test_classifier_teacher_immutability_stft_and_encoder_gradients(self):
        model = TFMClassifier(SmallTokenizer(), SmallEncoder(), freeze_encoder=True).train()
        before = {name: value.clone() for name, value in model.state_dict().items() if not name.startswith('classifier.')}
        self.assertFalse(model.tokenizer.training)
        self.assertFalse(model.encoder.training)
        x = torch.randn(2, 3, 400, requires_grad=True)
        ids = model.get_codebook_indices(x)
        wave = x.detach().reshape(6, 400)
        spectrum = torch.stft(wave, n_fft=200, hop_length=100, center=False,
            window=torch.hann_window(200), return_complex=True).abs()[:, :100]
        with torch.no_grad():
            _, expected_ids, _ = model.tokenizer.tokenize(spectrum, wave)
        torch.testing.assert_close(ids, expected_ids.reshape(2, 3, 3), rtol=0, atol=0)
        model(x).square().sum().backward()
        self.assertIsNone(x.grad)
        self.assertTrue(all(p.grad is None for p in model.tokenizer.parameters()))
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        for name, value in before.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        unfrozen = TFMClassifier(SmallTokenizer(), SmallEncoder()).train()
        unfrozen(x.detach()).square().sum().backward()
        self.assertIsNotNone(unfrozen.encoder.eeg_token_embedding.weight.grad)
        self.assertIsNone(unfrozen.encoder.classification_head.weight.grad)
        self.assertTrue(all(p.grad is None for p in unfrozen.tokenizer.parameters()))

    def test_complete_strict_loading_prefix_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            for cls in [SmallTokenizer, SmallEncoder]:
                state = cls().state_dict(); path = Path(directory)/'model.pt'
                torch.save(state, path); restored = cls.from_pretrained(path)
                for key, value in state.items():
                    torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
                torch.save({'module.'+key: value for key, value in state.items()}, path)
                cls.from_pretrained(path, prefix='module.')
                with self.assertRaises(ValueError): cls.from_pretrained(path)
                key = 'decoder.weight' if cls is SmallTokenizer else 'classification_head.weight'
                for value in [None, state[key].double(), state[key][..., :1], torch.full_like(state[key], float('nan'))]:
                    bad = dict(state)
                    if value is None: del bad[key]
                    else: bad[key] = value
                    torch.save(bad, path)
                    with self.assertRaises(ValueError): cls.from_pretrained(path)
                for bad in [dict(state, extra=torch.ones(1)), dict(model=state), dict(state, bad='object')]:
                    torch.save(bad, path)
                    with self.assertRaises(ValueError): cls.from_pretrained(path)
                path = Path(directory)/'model.safetensors'; save_file(state, path); cls.from_pretrained(path)
            # Both full reconstruction-head variants round-trip without skipping tensors.
            path = Path(directory)/'mlp.pt'; mlp = SmallTokenizer(reconstruction_head='mlp')
            torch.save(mlp.state_dict(), path)
            SmallTokenizer.from_pretrained(path, reconstruction_head='mlp')
            with self.assertRaises(ValueError): SmallTokenizer.from_pretrained(path)
            bad = SmallTokenizer().state_dict(); bad['quantizer.cluster_size'][0] = -1; torch.save(bad, path)
            with self.assertRaises(ValueError): SmallTokenizer.from_pretrained(path)
            bad = SmallEncoder().state_dict(); bad['index'][0] = 1; torch.save(bad, path)
            with self.assertRaises(ValueError): SmallEncoder.from_pretrained(path)

    def test_invalid_inputs(self):
        for cls in [SmallTokenizer, SmallEncoder]:
            with self.assertRaises(ValueError): cls(hid_channels=7)
            with self.assertRaises(ValueError): cls(num_tokens=0)
            with self.assertRaises(ValueError): cls.from_pretrained('missing', prefix=1)
        for kwargs in [dict(decay=1), dict(beta=float('nan')), dict(update_codebook=1), dict(reconstruction_head='other')]:
            with self.assertRaises(ValueError): SmallTokenizer(**kwargs)
        model = SmallTokenizer()
        for spectrum, temporal in [(torch.zeros(2, 99, 1), torch.zeros(2, 200)),
                                   (torch.zeros(2, 100, 2), torch.zeros(2, 200)),
                                   (torch.zeros(2, 100, 1).long(), torch.zeros(2, 200)),
                                   (torch.zeros(0, 100, 1), torch.zeros(0, 200))]:
            with self.assertRaises(ValueError): model(spectrum, temporal)
        encoder = SmallEncoder()
        for ids in [torch.zeros(1, 1, 1), torch.full((1, 1, 1), 24), torch.full((1, 1, 1), -1), torch.zeros(1, 17, 1).long()]:
            with self.assertRaises(ValueError): encoder(ids)
        for rows in [[0, 0], [-1, 2], [0, 16], [0.1, 2]]:
            with self.assertRaises(ValueError): encoder(torch.zeros(1, 2, 1).long(), rows)
        with self.assertRaises(ValueError): TFMClassifier(model, SmallEncoder(num_tokens=24))
        classifier = TFMClassifier(model, encoder)
        for x in [torch.zeros(1, 17, 200), torch.zeros(1, 1, 199), torch.zeros(1, 1, 200).long()]:
            with self.assertRaises(ValueError): classifier(x)


if __name__ == '__main__':
    unittest.main()
