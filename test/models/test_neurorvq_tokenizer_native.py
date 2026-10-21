"""Complete official 505-state tokenizer oracle, external license/deps required."""

import copy
from datetime import timedelta
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from torcheeg.models import NeuroRVQTokenizer


def load_example():
    path = (
        Path(__file__).resolve().parents[2]
        / 'examples'
        / 'examples_eegmat_neurorvq_tokenizer_training.py'
    )
    spec = importlib.util.spec_from_file_location('neurorvq_tokenizer_example', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _distributed_ema_worker(rank, rendezvous):
    """Full official tokenizer, deliberately constructed BEFORE the process group."""
    torch.set_num_threads(2)
    example = load_example()
    native, vocab = example.build_neurorvq_tokenizer(
        os.environ['TORCHEEG_NEURORVQ_BACKEND_ROOT']
    )
    model = NeuroRVQTokenizer.from_pretrained(
        os.environ['TORCHEEG_NEURORVQ_TOKENIZER_CHECKPOINT'],
        native,
        ['a1', 'fp1', 'fp2'],
        vocab,
    ).train()
    layers = model._quantizers()
    before = [layer.cluster_size.clone() for layer in layers]
    torch.distributed.init_process_group(
        'gloo',
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        values = torch.sin(torch.arange(1200).float() / (31 + rank)).reshape(1, 3, 400)
        with torch.no_grad():
            indices = model.get_codebook_indices(values)
        gathered = [torch.empty_like(indices) for _ in range(2)]
        torch.distributed.all_gather(gathered, indices)
        global_indices = torch.cat([value.flatten(1) for value in gathered], dim=1)
        digests = []
        for index, (layer, old) in enumerate(zip(layers, before)):
            counts = torch.bincount(global_indices[index], minlength=8192).to(old)
            expected = old.mul(layer.decay).add(counts, alpha=1 - layer.decay)
            torch.testing.assert_close(layer.cluster_size, expected, rtol=0, atol=0)
            digests.append(
                hashlib.sha256(
                    layer.embedding.weight.detach().numpy().tobytes()
                ).hexdigest()
            )
        hashes = [None, None]
        torch.distributed.all_gather_object(hashes, digests)
        assert hashes[0] == hashes[1], (
            'All 32 updated EMA codebooks must agree across ranks.'
        )
    finally:
        torch.distributed.destroy_process_group()


@unittest.skipUnless(
    os.environ.get('TORCHEEG_NEURORVQ_BACKEND_ROOT')
    and os.environ.get('TORCHEEG_NEURORVQ_TOKENIZER_CHECKPOINT'),
    'Explicit separately licensed complete tokenizer source/checkpoint required.',
)
class TestNeuroRVQTokenizerNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.example = load_example()
        native, cls.vocab = cls.example.build_neurorvq_tokenizer(
            os.environ['TORCHEEG_NEURORVQ_BACKEND_ROOT']
        )
        cls.model = NeuroRVQTokenizer.from_pretrained(
            os.environ['TORCHEEG_NEURORVQ_TOKENIZER_CHECKPOINT'],
            native,
            ['a1', 'fp1', 'fp2'],
            cls.vocab,
        )
        cls.initial = copy.deepcopy(cls.model.state_dict())
        cls.reference = copy.deepcopy(native)
        cls.values = torch.sin(torch.arange(1200).float() / 31).reshape(1, 3, 400)

    def setUp(self):
        self.model.load_state_dict(self.initial, strict=True)
        self.reference.load_state_dict(
            {
                key[8:]: value
                for key, value in self.initial.items()
                if key.startswith('backend.')
            },
            strict=True,
        )

    def test_complete_native_losses_gradients_ema_and_output(self):
        self.assertEqual(len(self.model.backend.state_dict()), 505)
        self.assertEqual(
            sum(v.numel() for v in self.model.backend.state_dict().values()), 75988624
        )
        self.assertEqual(
            sum(p.numel() for p in self.model.parameters() if p.requires_grad), 8355440
        )
        temporal = torch.tensor([[254, 255] * 3])
        spatial = torch.tensor([[0, 0, 55, 55, 56, 56]])
        for training in [False, True]:
            self.setUp()
            self.model.train(training)
            self.reference.train(training)
            self.model.zero_grad(set_to_none=True)
            self.reference.zero_grad(set_to_none=True)
            left = self.values.clone().requires_grad_()
            right = self.values.clone().requires_grad_()
            captured = {}
            encode = self.reference.encode

            def capture_encode(*args, **kwargs):
                result = encode(*args, **kwargs)
                captured['code_loss'] = result[2]
                captured['indices'] = torch.stack(
                    [value.reshape(8, 1, 3, 2) for value in result[1]]
                ).reshape(32, 1, 3, 2)
                return result

            def capture_loss(name, method):
                def wrapped(*args, **kwargs):
                    result = method(*args, **kwargs)
                    captured[name] = result
                    return result

                return wrapped

            with patch.object(self.reference, 'encode', capture_encode), patch.object(
                self.reference,
                'calculate_rec_loss',
                capture_loss('amplitude_loss', self.reference.calculate_rec_loss),
            ), patch.object(
                self.reference,
                'calculate_phase_loss',
                capture_loss('phase_loss', self.reference.calculate_phase_loss),
            ), patch.object(
                self.reference,
                'calculate_signal_rec_loss',
                capture_loss('signal_loss', self.reference.calculate_signal_rec_loss),
            ):
                original, reconstruction = self.reference(right, temporal, spatial)
            actual = self.model(left, return_loss=True)
            torch.testing.assert_close(
                actual['original'].reshape(1, 6, 200), original, rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual['reconstruction'].reshape(1, 6, 200),
                reconstruction,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                actual['indices'], captured.pop('indices'), rtol=0, atol=0
            )
            for key, value in captured.items():
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0, msg=key)
            # Match the author's expression, not callback insertion order:
            # the source computes signal loss before amplitude and phase.
            expected_loss = (
                captured['code_loss']
                + captured['amplitude_loss']
                + captured['phase_loss']
                + captured['signal_loss']
            )
            torch.testing.assert_close(actual['loss'], expected_loss, rtol=0, atol=0)
            actual['loss'].backward()
            expected_loss.backward()
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
            used = 0
            for name, parameter in self.model.backend.named_parameters():
                reference = dict(self.reference.named_parameters())[name]
                self.assertEqual(parameter.grad is None, reference.grad is None, name)
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    torch.testing.assert_close(
                        parameter.grad, reference.grad, rtol=0, atol=0, msg=name
                    )
                    used += 1
            self.assertGreater(used, 200)
            for name, state in self.model.backend.state_dict().items():
                expected = self.reference.state_dict()[name]
                if (
                    not training
                    and name.endswith('.cluster_size')
                    and '.embedding.' not in name
                ):
                    expected = self.initial['backend.' + name]
                torch.testing.assert_close(state, expected, rtol=0, atol=0, msg=name)

    def test_token_order_readonly_eval_native_usage_and_validation(self):
        self.model.eval()
        self.reference.eval()
        patches, temporal, spatial = self.model._inputs(self.values)
        with torch.no_grad():
            actual = self.model.get_tokens(self.values)
            native = self.reference.get_tokens(patches, temporal, spatial)
        torch.testing.assert_close(actual['token'], native['token'], rtol=0, atol=0)
        for branch in range(4):
            for residual in range(8):
                torch.testing.assert_close(
                    actual['indices'][branch * 8 + residual].flatten(1),
                    native['token'][branch, residual],
                    rtol=0,
                    atol=0,
                )
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(
                value, self.initial[name], rtol=0, atol=0, msg=name
            )

        def interrupted_encode(*args, **kwargs):
            self.model.backend.quantize_1.layers[0].cluster_size.add_(1)
            raise RuntimeError('interrupted native encoder')

        with patch.object(self.model.backend, 'encode', interrupted_encode):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                self.model.get_tokens(self.values)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(
                value, self.initial[name], rtol=0, atol=0, msg=name
            )
        tracked = NeuroRVQTokenizer(
            self.reference, ['a1', 'fp1', 'fp2'], self.vocab, track_usage=True
        ).eval()
        before = {name: value.clone() for name, value in self.reference.named_buffers()}
        with torch.no_grad():
            tracked.get_codebook_indices(self.values)
        changed = [
            name
            for name, value in self.reference.named_buffers()
            if not torch.equal(value, before[name])
        ]
        self.assertEqual(len(changed), 32)
        self.assertTrue(all(name.endswith('.cluster_size') for name in changed))
        with self.assertRaisesRegex(RuntimeError, 'policy'):
            tracked.load_state_dict(self.initial, strict=True)
        self.model.backend.quantize_1.layers[0].embedding.initted.zero_()
        with self.assertRaisesRegex(ValueError, 'initialized'):
            self.model.get_tokens(self.values)
        for bad in [
            torch.ones(1, 2, 400),
            torch.ones(1, 3, 399),
            torch.ones(0, 3, 400),
            torch.ones(1, 3, 200, dtype=torch.long),
        ]:
            with self.assertRaises(ValueError):
                self.model.get_tokens(bad)

    def test_late_process_group_synchronizes_all_32_ema_codebooks(self):
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_gloo_available()
        ):
            self.skipTest('The full two-rank EMA oracle needs Gloo.')
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / 'rendezvous').as_uri()
            torch.multiprocessing.spawn(
                _distributed_ema_worker, args=(rendezvous,), nprocs=2, join=True
            )

    def test_strict_native_tensor_loading_and_prefix(self):
        state = {
            key[8:]: value
            for key, value in self.initial.items()
            if key.startswith('backend.')
        }
        key = 'encoder.cls_token'
        invalid = [
            [],
            {'state_dict': state},
            {**state, 'extra': torch.ones(1)},
            {name: value for name, value in state.items() if name != key},
            {**state, key: torch.ones(1)},
            {**state, key: state[key].double()},
            {**state, key: torch.full_like(state[key], float('nan'))},
        ]
        for values in invalid:
            with self.subTest(kind=type(values).__name__):
                with patch('torch.load', return_value=values):
                    with self.assertRaises(ValueError):
                        NeuroRVQTokenizer.from_pretrained(
                            'bad.pt', self.reference, ['a1', 'fp1', 'fp2'], self.vocab
                        )
        with patch(
            'torch.load',
            return_value={'module.' + name: value for name, value in state.items()},
        ):
            loaded = NeuroRVQTokenizer.from_pretrained(
                'valid.pt',
                self.reference,
                ['a1', 'fp1', 'fp2'],
                self.vocab,
                prefix='module.',
            )
        for name, value in loaded.backend.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        with patch('torch.load', return_value=state):
            with self.assertRaisesRegex(ValueError, 'prefix'):
                NeuroRVQTokenizer.from_pretrained(
                    'bad.pt',
                    self.reference,
                    ['a1', 'fp1', 'fp2'],
                    self.vocab,
                    prefix='module.',
                )

    def test_full_optimizer_rng_ema_resume_and_metadata(self):
        model = self.model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        def step(values):
            optimizer.zero_grad(set_to_none=True)
            result = model.forward_pretraining(values)
            result['loss'].backward()
            optimizer.step()
            return result['loss'].detach().clone()

        torch.manual_seed(42)
        step(self.values + torch.randn_like(self.values) * 0.01)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'resume.pt'
            torch.save(
                dict(
                    state=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    rng=torch.get_rng_state(),
                ),
                path,
            )
            expected_loss = step(self.values + torch.randn_like(self.values) * 0.01)
            expected = copy.deepcopy(model.state_dict())
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            payload = torch.load(path, weights_only=True, map_location='cpu')
            model.load_state_dict(payload['state'], strict=True)
            optimizer.load_state_dict(payload['optimizer'])
            torch.set_rng_state(payload['rng'])
            actual_loss = step(self.values + torch.randn_like(self.values) * 0.01)
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(
                    value, expected[name], rtol=0, atol=0, msg=name
                )
            for key, state in optimizer.state_dict()['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(
                        value, expected_optimizer['state'][key][name], rtol=0, atol=0
                    )
        altered = copy.deepcopy(self.initial)
        altered['channel_fingerprint'][0] ^= 1
        with self.assertRaisesRegex(RuntimeError, 'metadata'):
            model.load_state_dict(altered, strict=True)


if __name__ == '__main__':
    unittest.main()
