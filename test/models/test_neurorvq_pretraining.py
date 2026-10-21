"""Protocol-only test doubles; complete native networks are tested separately."""

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from torcheeg.models import NeuroRVQ, NeuroRVQTokenizer, NeuroRVQMaskedTokenPretrainer
from torcheeg.models.transformer.neurorvq import _NeuroRVQInput


def load_example():
    path = (
        Path(__file__).resolve().parents[2]
        / 'examples'
        / 'examples_eegmat_neurorvq_mtp_training.py'
    )
    spec = importlib.util.spec_from_file_location('neurorvq_mtp_example', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ProtocolLogits(nn.Module):
    """A test-only score table, not a NeuroRVQ implementation or checkpoint."""

    def __init__(self):
        super().__init__()
        self.scores = nn.Parameter(
            torch.sin(torch.arange(32 * 8192).reshape(32, 8192) * 0.01) * 0.05
        )
        self.register_buffer('classes', torch.linspace(-1, 1, 8192))

    def forward(self, x, temporal, spatial, bool_masked_pos, use_for_pretraining):
        features = x.mean(-1).flatten(1)

        def scores(mask, sign):
            return (
                self.scores[:, None, :]
                + sign * features[mask][None, :, None] * self.classes
            )

        return scores(bool_masked_pos, 1), scores(~bool_masked_pos, -1)


class _ProtocolBackbone(NeuroRVQ):
    """Bypass full-architecture construction ONLY for wrapper-protocol tests."""

    def __init__(self):
        _NeuroRVQInput.__init__(self, ['fp1', 'fp2'], ['fp1', 'fp2'])
        self.backend = _ProtocolLogits()


class _ProtocolTokenizer(NeuroRVQTokenizer):
    """Known indices, not a tokenizer substitute or native-model oracle."""

    def __init__(self):
        _NeuroRVQInput.__init__(self, ['fp1', 'fp2'], ['fp1', 'fp2'])
        self.track_usage = False
        self.parameter = nn.Parameter(torch.ones(1))
        self.register_buffer('seen', torch.zeros(1))

    def get_codebook_indices(self, x):
        patches, _, _ = self._inputs(x)
        self.called_with_grad = torch.is_grad_enabled()
        self.called_training = self.training
        shape = (32,) + tuple(patches.shape[:3])
        return (
            torch.arange(torch.tensor(shape).prod().item(), device=x.device)
            .reshape(shape)
            .remainder(8192)
        )


class TestNeuroRVQPretraining(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.backbone = _ProtocolBackbone()
        self.teacher = _ProtocolTokenizer()
        self.model = NeuroRVQMaskedTokenPretrainer(self.backbone, self.teacher)
        self.values = torch.linspace(-2, 3, 2 * 2 * 600).reshape(2, 2, 600)
        self.mask = torch.tensor(
            [
                [[True, False, False], [False, True, False]],
                [[False, False, False], [False, False, False]],
            ]
        )

    def test_ordered_targets_and_independent_partition_means(self):
        values = self.values.clone().requires_grad_()
        result = self.model(values, self.mask)
        targets = torch.arange(32 * 12).reshape(32, 2, 2, 3)
        torch.testing.assert_close(result['targets'], targets, rtol=0, atol=0)
        expected = []
        for logits, selected in [
            (result['logits'], self.mask),
            (result['symmetric_logits'], ~self.mask),
        ]:
            labels = targets.flatten(1)[:, selected.flatten()]
            # Independent high-precision log-probability definition of CE.
            logp = logits.double().log_softmax(-1)
            expected.append(-logp.gather(-1, labels.unsqueeze(-1)).mean())
        torch.testing.assert_close(
            result['masked_loss'].double(), expected[0], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            result['symmetric_loss'].double(), expected[1], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            result['loss'].double(), sum(expected), rtol=1e-6, atol=1e-6
        )
        self.assertEqual(result['logits'].shape, (32, 2, 8192))
        self.assertEqual(result['symmetric_logits'].shape, (32, 10, 8192))
        result['loss'].backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertGreater(values.grad.abs().sum().item(), 0)
        self.assertTrue((self.backbone.backend.scores.grad.abs().sum(-1) > 0).all())
        self.assertIsNone(self.teacher.parameter.grad)

    def test_teacher_stays_frozen_and_all_its_states_stay_unchanged(self):
        before = copy.deepcopy(self.teacher.state_dict())
        original = self.backbone.backend.scores.detach().clone()
        optimizer = torch.optim.AdamW(self.backbone.parameters(), lr=0.01)
        self.model.train()
        self.assertTrue(self.backbone.training)
        self.assertFalse(self.teacher.training)
        self.assertFalse(self.teacher.parameter.requires_grad)
        self.teacher.train()  # A direct accidental mode change is repaired at call time.
        result = self.model(self.values, self.mask)
        self.assertFalse(self.teacher.called_with_grad)
        self.assertFalse(self.teacher.called_training)
        result['loss'].backward()
        optimizer.step()
        self.assertFalse(torch.equal(self.backbone.backend.scores, original))
        for name, value in self.teacher.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.model.eval()
        self.assertFalse(self.backbone.training)
        self.assertFalse(self.teacher.training)

    def test_constructor_rejects_wrong_metadata_tracking_and_shared_parameters(self):
        with self.assertRaises(ValueError):
            NeuroRVQMaskedTokenPretrainer(nn.Identity(), self.teacher)
        with self.assertRaises(ValueError):
            NeuroRVQMaskedTokenPretrainer(self.backbone, nn.Identity())
        for name, value in [
            ('channel_names', ('fp2', 'fp1')),
            ('channel_vocab', ('fp2', 'fp1')),
            ('sampling_rate', 256),
            ('patch_size', 100),
            ('max_patches', 128),
        ]:
            teacher = _ProtocolTokenizer()
            setattr(teacher, name, value)
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, 'metadata'):
                    NeuroRVQMaskedTokenPretrainer(self.backbone, teacher)
        teacher = _ProtocolTokenizer()
        teacher.track_usage = True
        with self.assertRaisesRegex(ValueError, 'read-only'):
            NeuroRVQMaskedTokenPretrainer(self.backbone, teacher)
        teacher = _ProtocolTokenizer()
        teacher.shared = self.backbone.backend.scores
        with self.assertRaisesRegex(ValueError, 'share parameters'):
            NeuroRVQMaskedTokenPretrainer(self.backbone, teacher)
        self.assertTrue(self.backbone.backend.scores.requires_grad)

    def test_mask_input_target_and_logit_contracts(self):
        for mask in [
            torch.zeros_like(self.mask),
            torch.ones_like(self.mask),
            self.mask.float(),
            self.mask[:, :, :2],
        ]:
            with self.assertRaises(ValueError):
                self.model(self.values, mask)
        for values in [self.values[:, :1], self.values[:, :, :599], self.values.long()]:
            with self.assertRaises(ValueError):
                self.model(values, self.mask)
        for targets in [
            torch.zeros(32, 2, 2, 3, dtype=torch.bool),
            torch.zeros(31, 2, 2, 3, dtype=torch.long),
            torch.full((32, 2, 2, 3), -1, dtype=torch.long),
            torch.full((32, 2, 2, 3), 8192, dtype=torch.long),
        ]:
            with patch.object(
                self.teacher, 'get_codebook_indices', return_value=targets
            ):
                with self.assertRaisesRegex(ValueError, 'target streams'):
                    self.model(self.values, self.mask)
        logits = self.backbone.forward_masked(self.values, self.mask)
        with patch.object(
            self.backbone,
            'forward_masked',
            return_value=(logits[0][..., :8], logits[1]),
        ):
            with self.assertRaisesRegex(ValueError, 'logit tensors'):
                self.model(self.values, self.mask)

    def test_objective_guard_and_complete_wrapper_restore(self):
        result = self.model(self.values, self.mask)
        state = copy.deepcopy(self.model.state_dict())
        restored = NeuroRVQMaskedTokenPretrainer(
            _ProtocolBackbone(), _ProtocolTokenizer()
        )
        restored.to('meta').to_empty(device='cpu')
        restored.load_state_dict(state, strict=True)
        actual = restored(self.values, self.mask)
        for name in result:
            torch.testing.assert_close(actual[name], result[name], rtol=0, atol=0)
        state['objective_fingerprint'][0] ^= 1
        with self.assertRaisesRegex(RuntimeError, 'reduction'):
            restored.load_state_dict(state, strict=True)

    def test_example_mask_policy_rng_and_shapes(self):
        example = load_example()
        generator = torch.Generator().manual_seed(17)
        values = torch.zeros(3, 3, 600)
        state = generator.get_state()
        mask = example.make_mask(values, generator)
        self.assertEqual(mask.shape, (3, 3, 3))
        self.assertTrue((mask.flatten(1).sum(1) == 4).all())
        self.assertFalse(torch.equal(generator.get_state(), state))
        generator.set_state(state)
        torch.testing.assert_close(
            example.make_mask(values, generator), mask, rtol=0, atol=0
        )
        for bad in [torch.zeros(1, 1, 200), torch.zeros(1, 2, 201)]:
            with self.assertRaises(ValueError):
                example.make_mask(bad, generator)

    def test_example_cli_requires_pair_or_resume_and_new_output(self):
        example = load_example()
        base = [
            'example',
            '--root-path',
            'missing-data',
            '--backend-root',
            'missing-source',
            '--output-path',
            'new.pt',
        ]
        for extra in [
            [],
            ['--foundation-checkpoint', 'only-one'],
            ['--tokenizer-checkpoint', 'only-one'],
            [
                '--resume',
                'resume.pt',
                '--foundation-checkpoint',
                'a',
                '--tokenizer-checkpoint',
                'b',
            ],
            ['--resume', 'resume.pt', '--steps', '0'],
        ]:
            with patch('sys.argv', base + extra):
                with self.assertRaises(SystemExit):
                    example.main()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'already.pt'
            path.write_text('preserve')
            with patch('sys.argv', base[:-1] + [str(path), '--resume', 'resume.pt']):
                with self.assertRaises(SystemExit):
                    example.main()
            self.assertEqual(path.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
