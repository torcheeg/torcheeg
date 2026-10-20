import argparse
from contextlib import contextmanager
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from safetensors.torch import save_file

from torcheeg.models import CodeBrain, CodeBrainTokenizer
from torcheeg.models.transformer.codebrain_tokenizer import _contrastive_loss, _NormQuantizer


SMALL = dict(encoder_depth=1, decoder_depth=1, code_dim=8, num_time_tokens=7, num_frequency_tokens=9)


class TinyTokenizer(CodeBrainTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**dict(SMALL, **kwargs))


def reference_state(state):
    result = {}
    for name, value in state.items():
        if name.endswith('num_batches_tracked') or name.endswith('running_mean') or 'cluster_size' in name:
            result[name] = torch.zeros_like(value)
        elif name.endswith('running_var') or name.endswith('initted') or (name.endswith('weight') and value.ndim == 1):
            result[name] = torch.ones_like(value)
        elif name.endswith('bias'):
            result[name] = torch.zeros_like(value)
        else:
            offset = int(hashlib.sha256(name.encode()).hexdigest()[:6], 16) % 97
            values = torch.sin(torch.arange(value.numel(), dtype=torch.float64) / 17 + offset) * .02
            result[name] = values.reshape_as(value).to(value.dtype)
    for domain in ['t', 'f']:
        name = f'quantize_{domain}.embedding.weight'
        result[name] = torch.nn.functional.normalize(result[name], dim=-1)
        result[f'quantize_{domain}.embedding.embed_avg'] = result[name].clone()
    return result


@contextmanager
def controlled_dropout():
    def forward(module, x):
        if not module.training or module.p == 0:
            return x
        mask = (torch.arange(x.numel(), device=x.device).reshape_as(x) % 10 >= int(module.p * 10)).to(x.dtype)
        return x * mask / (1-module.p)
    with patch.object(torch.nn.Dropout, 'forward', forward):
        yield


class _MaliciousMetadata:
    def __init__(self, target):
        self.target = target

    def __reduce__(self):
        return os.system, ('touch ' + self.target,)


def _distributed_codebook_worker(rank, directory):
    root = Path(directory)
    torch.manual_seed(rank+1)
    # Frameworks may construct the module before setting up the process group.
    quantizer = _NormQuantizer(7, 8, .9, True, True, True).train()
    distributed.init_process_group('gloo', init_method=(root/'rendezvous').as_uri(),
                                   rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        quantizer(torch.randn(1, 8, 2, 2)+rank*.5)
        torch.save(dict(weight=quantizer.embedding.weight.detach().clone(),
                        initial_counts=quantizer.embedding.cluster_size.detach().clone(),
                        usage=quantizer.cluster_size.clone()), root/f'rank-{rank}.pt')
    finally:
        distributed.destroy_process_group()


class TestCodeBrainTokenizer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.eeg = torch.randn(2, 3, 4, 200)

    def test_native_numerical_losses_and_every_gradient(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/codebrain_tokenizer_reference.json').read_text())
        model = TinyTokenizer(quantize_kmeans_init=False, contrastive_mode='native')
        model.load_state_dict(reference_state(model.state_dict()), strict=True)
        for label in ['eval_no_grad', 'eval_grad', 'train_grad']:
            case = fixture['cases'][label]
            model.train(label == 'train_grad'); model.zero_grad(set_to_none=True)
            x = (torch.sin(torch.arange(2*3*4*200)/17)+torch.cos(torch.arange(2*3*4*200)/31)).reshape(2, 3, 4, 200)
            x.requires_grad_(label != 'eval_no_grad')
            with torch.set_grad_enabled(x.requires_grad), controlled_dropout():
                loss, logs = model(x)
            torch.testing.assert_close(loss, torch.tensor(case['loss']), rtol=3e-5, atol=3e-6)
            for name, value in logs.items():
                torch.testing.assert_close(value, torch.tensor(case['logs'][name]), rtol=3e-5, atol=3e-6)
            if x.requires_grad:
                loss.backward()
                torch.testing.assert_close(x.grad, torch.tensor(case['input_gradient']), rtol=5e-4, atol=1e-5)
                for name, parameter in model.named_parameters():
                    with self.subTest(case=label, parameter=name):
                        expected = case['parameter_gradients'][name]
                        if expected is None:
                            self.assertIsNone(parameter.grad)
                        else:
                            gradient = parameter.grad.double().flatten()
                            torch.testing.assert_close(gradient[:8], torch.tensor(expected['first'], dtype=torch.float64), rtol=5e-4, atol=1e-5)
                            self.assertAlmostEqual(gradient.sum().item(), expected['sum'], delta=5e-5)
                            self.assertAlmostEqual(gradient.norm().item(), expected['norm'], delta=5e-5)
        model.eval()
        with torch.no_grad():
            encoded = model.encode(x.detach())
            expected = fixture['outputs']
            torch.testing.assert_close(encoded[1], torch.tensor(expected['time_ids']), rtol=0, atol=0)
            torch.testing.assert_close(encoded[4], torch.tensor(expected['frequency_ids']), rtol=0, atol=0)
            torch.testing.assert_close(encoded[0], torch.tensor(expected['quantized_time']), rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(encoded[3], torch.tensor(expected['quantized_frequency']), rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(model.decode_t(encoded[0]), torch.tensor(expected['raw']), rtol=3e-5, atol=3e-6)
            amplitude, phase = model.decode_f(encoded[3])
            torch.testing.assert_close(amplitude, torch.tensor(expected['amplitude']), rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(phase, torch.tensor(expected['phase']), rtol=3e-5, atol=3e-6)

    def test_corrected_infonce_excludes_positive_pair(self):
        vectors = torch.eye(2)
        loss = _contrastive_loss(vectors, vectors, 'infonce')
        self.assertAlmostEqual(loss.item(), math.log1p(2*math.exp(-10)), delta=1e-6)
        native = _contrastive_loss(vectors, vectors, 'native')
        self.assertAlmostEqual(native.item(), math.log(2+2*math.exp(-10)), delta=1e-6)
        one = vectors[:1]
        self.assertEqual(_contrastive_loss(one, one, 'infonce').item(), 0)
        self.assertAlmostEqual(_contrastive_loss(one, one, 'native').item(), math.log(2), delta=1e-6)

    def test_single_patch_and_constant_targets_are_finite(self):
        model = TinyTokenizer(quantize_kmeans_init=False, track_usage=False)
        x = torch.zeros(1, 2, 1, 200, requires_grad=True)
        loss, logs = model(x, contrastive=False)
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(logs['train/cl_loss'].item(), 0)
        model.eval()
        with torch.no_grad():
            encoded = model.encode(x)
            self.assertEqual(model.decode_t(encoded[0]).shape, (1, 2, 200))
            self.assertEqual(model.decode_f(encoded[3])[0].shape, (1, 2, 200))

    def test_eval_batch_isolation_permutation_and_fixed_rows(self):
        model = TinyTokenizer(quantize_kmeans_init=False, track_usage=False).eval()
        state = {name: value.clone() for name, value in model.state_dict().items()}
        x = self.eeg.clone().requires_grad_()
        encoded = model.encode(x, [2, 4, 6], contrastive=False)
        raw = model.decode_t(encoded[0], [2, 4, 6]).reshape_as(x)
        separate = []
        for index in range(2):
            result = model.encode(x[index:index+1], [2, 4, 6], contrastive=False)
            separate.append(model.decode_t(result[0], [2, 4, 6]).reshape_as(x[index:index+1]))
        torch.testing.assert_close(raw, torch.cat(separate), rtol=3e-5, atol=3e-6)
        raw[0].square().sum().backward()
        torch.testing.assert_close(x.grad[1], torch.zeros_like(x.grad[1]))
        with torch.no_grad():
            original = model.get_tokens(self.eeg, [2, 4, 6])
            reordered = model.get_tokens(self.eeg[:, [2, 0, 1]], [6, 2, 4])
            for key in ['token_t', 'token_f']:
                torch.testing.assert_close(reordered[key].reshape(2, 3, 4), original[key].reshape(2, 3, 4)[:, [2, 0, 1]])
            default = model.get_tokens(self.eeg)
            explicit = model.get_tokens(self.eeg, torch.tensor([1, 2, 3]))
            torch.testing.assert_close(default['quantize_t'], explicit['quantize_t'])
        for name, value in state.items(): torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

    def test_codebook_ema_usage_and_optimizer(self):
        model = TinyTokenizer(quantize_kmeans_init=False)
        before = model.quantize_t.embedding.weight.detach().clone()
        raw_weight = model.decode_task_layer_temp[-1].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        loss, _ = model(self.eeg); loss.backward(); optimizer.step()
        self.assertFalse(torch.equal(before, model.quantize_t.embedding.weight))
        self.assertFalse(torch.equal(raw_weight, model.decode_task_layer_temp[-1].weight))
        self.assertIsNone(model.quantize_t.embedding.weight.grad)
        torch.testing.assert_close(model.quantize_t.embedding.weight.norm(dim=-1), torch.ones(7))
        model.eval(); before = model.quantize_t.embedding.weight.detach().clone()
        count = model.quantize_t.cluster_size.clone()
        with torch.no_grad(): model(self.eeg)
        torch.testing.assert_close(model.quantize_t.embedding.weight, before, rtol=0, atol=0)
        self.assertFalse(torch.equal(count, model.quantize_t.cluster_size))

    def test_two_stage_teacher_targets_and_encoder_update(self):
        teacher = TinyTokenizer(quantize_kmeans_init=False, update_codebook=False, track_usage=False).eval()
        encoder = CodeBrain(hid_channels=8, depth=2, max_length=16, num_time_tokens=7, num_frequency_tokens=9)
        with torch.no_grad():
            time_ids, frequency_ids = teacher.get_codebook_indices(self.eeg)
        mask = torch.zeros(self.eeg.shape[:-1], dtype=torch.bool); mask[:, :, 0] = True
        time, frequency = encoder.predict_tokens(self.eeg, mask)
        loss = torch.nn.functional.cross_entropy(time, time_ids.reshape_as(mask)[mask])
        loss = loss + torch.nn.functional.cross_entropy(frequency, frequency_ids.reshape_as(mask)[mask])
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        self.assertGreater(encoder.lm_head_t.weight.grad.abs().sum().item(), 0)
        self.assertGreater(encoder.lm_head_f.weight.grad.abs().sum().item(), 0)

    def test_kmeans_initialization_and_restore_flags(self):
        model = TinyTokenizer(track_usage=False).eval()
        self.assertFalse(model.quantize_t.embedding._initialized)
        with torch.no_grad(): model.get_tokens(self.eeg)
        self.assertTrue(model.quantize_t.embedding._initialized)
        restored = TinyTokenizer(track_usage=False).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(restored.quantize_t.embedding._initialized)
        with torch.no_grad():
            first = model.get_codebook_indices(self.eeg)
            second = restored.get_codebook_indices(self.eeg)
        for a, b in zip(first, second): torch.testing.assert_close(a, b, rtol=0, atol=0)

    @unittest.skipUnless(distributed.is_available() and distributed.is_gloo_available(), 'CPU Gloo is unavailable.')
    def test_distributed_initialization_and_global_ema(self):
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.spawn(_distributed_codebook_worker, args=(directory,), nprocs=2, join=False)
            deadline = time.monotonic()+60
            while not context.join(timeout=1):
                if time.monotonic() >= deadline:
                    for process in context.processes: process.terminate()
                    self.fail('Distributed codebook test exceeded60seconds.')
            root = Path(directory)
            first = torch.load(root/'rank-0.pt', weights_only=True)
            second = torch.load(root/'rank-1.pt', weights_only=True)
            for key in first: torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
            self.assertAlmostEqual(first['initial_counts'].sum().item(), 4)
            self.assertAlmostEqual(first['usage'].sum().item(), .8, delta=1e-6)
            torch.testing.assert_close(first['weight'].norm(dim=-1), torch.ones(7))

    def test_safe_loader_plain_safetensors_metadata_and_unknown_globals(self):
        model = TinyTokenizer(quantize_kmeans_init=False)
        state = model.state_dict()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root/'model.pt'
            torch.save(state, path)
            loaded = TinyTokenizer.from_pretrained(path)
            for name, value in state.items(): torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
            safe = root/'model.safetensors'; save_file(state, str(safe))
            TinyTokenizer.from_pretrained(safe)
            if hasattr(torch.serialization, 'safe_globals'):
                before = set(torch.serialization.get_safe_globals())
                torch.save(dict(model=state, args=argparse.Namespace(path='not executed'), score=np.float64(.25)), path)
                with patch('argparse.Namespace', side_effect=AssertionError('Real Namespace constructor executed')):
                    TinyTokenizer.from_pretrained(path)
                self.assertEqual(set(torch.serialization.get_safe_globals()), before)
            marker = root/'must_not_exist'
            torch.save(dict(model=state, metadata=_MaliciousMetadata(str(marker))), path)
            with self.assertRaises(Exception): TinyTokenizer.from_pretrained(path)
            self.assertFalse(marker.exists())

    def test_strict_loader_rejects_corruption_and_uninitialized_codebooks(self):
        state = TinyTokenizer(quantize_kmeans_init=False).state_dict()
        key = 'decoder_f.patch_embed.proj.weight'
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model.pt'
            for value in [None, state[key][:1], state[key].double(), torch.full_like(state[key], float('nan'))]:
                bad = dict(state)
                if value is None: del bad[key]
                else: bad[key] = value
                torch.save(bad, path)
                with self.assertRaises(ValueError): TinyTokenizer.from_pretrained(path)
            for bad in [dict(state, unexpected=torch.zeros(1)), TinyTokenizer().state_dict(), {'wrong': 'object'}]:
                torch.save(bad, path)
                with self.assertRaises(ValueError): TinyTokenizer.from_pretrained(path)
            torch.save({'teacher.'+name: value for name, value in state.items()}, path)
            TinyTokenizer.from_pretrained(path, prefix='teacher.')

    def test_invalid_input_and_configuration(self):
        for kwargs in [dict(code_dim=0), dict(encoder_depth=False), dict(decay=1), dict(track_usage=1), dict(contrastive_mode='bad')]:
            with self.assertRaises(ValueError): TinyTokenizer(**kwargs)
        model = TinyTokenizer(quantize_kmeans_init=False)
        for x in [self.eeg.long(), self.eeg[..., :199], self.eeg[:0], torch.zeros(1, 1, 61, 200)]:
            with self.assertRaises(ValueError): model(x)
        for ids in [[1, 1, 2], [0, 2, 3], [1, 2, 129], ['a', 'b', 'c'], torch.zeros(3), [1, 2]]:
            with self.assertRaises(ValueError): model(self.eeg, ids)
        with self.assertRaises(ValueError): model(self.eeg[:, :, :3])
        with self.assertRaises(ValueError): model.encode(self.eeg, contrastive=1)
        with self.assertRaises(ValueError): model.decode_t(torch.zeros(1, 4, 3, 2))
        with self.assertRaises(ValueError): TinyTokenizer.from_pretrained('missing', update_codebook=1)


if __name__ == '__main__':
    unittest.main()
