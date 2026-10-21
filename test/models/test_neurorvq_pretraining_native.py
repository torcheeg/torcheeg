"""Complete official 505-state teacher and 330-state backbone, no reduced nets."""

import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from torcheeg.models import NeuroRVQ, NeuroRVQTokenizer, NeuroRVQMaskedTokenPretrainer
from test.models.test_neurorvq_pretraining import load_example


def digest(value):
    """Record every tensor exactly without keeping a second full optimizer copy."""
    if isinstance(value, torch.Tensor):
        return (
            str(value.dtype),
            tuple(value.shape),
            hashlib.sha256(
                value.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
        )
    if isinstance(value, dict):
        return {key: digest(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [digest(item) for item in value]
    return value


@unittest.skipUnless(
    os.environ.get('TORCHEEG_NEURORVQ_BACKEND_ROOT')
    and os.environ.get('TORCHEEG_NEURORVQ_CHECKPOINT')
    and os.environ.get('TORCHEEG_NEURORVQ_TOKENIZER_CHECKPOINT'),
    'Explicit complete separately licensed native source and both official checkpoints required.',
)
class TestNeuroRVQPretrainingNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.example = load_example()
        names = ['a1', 'fp1', 'fp2']
        root = os.environ['TORCHEEG_NEURORVQ_BACKEND_ROOT']
        teacher_backend, vocab = cls.example.build_neurorvq_tokenizer(root)
        backbone_backend, config = cls.example.build_neurorvq_backend(root, names)
        if config['channel_vocab'] != vocab:
            raise ValueError('Original teacher/backbone vocabulary order differs.')
        backbone = NeuroRVQ.from_pretrained(
            os.environ['TORCHEEG_NEURORVQ_CHECKPOINT'], backbone_backend, names, vocab
        )
        teacher = NeuroRVQTokenizer.from_pretrained(
            os.environ['TORCHEEG_NEURORVQ_TOKENIZER_CHECKPOINT'],
            teacher_backend,
            names,
            vocab,
        )
        cls.model = NeuroRVQMaskedTokenPretrainer(backbone, teacher)
        cls.reference_backbone = copy.deepcopy(backbone_backend)
        cls.reference_teacher = copy.deepcopy(teacher_backend).eval()
        cls.original_backbone = torch.load(
            os.environ['TORCHEEG_NEURORVQ_CHECKPOINT'],
            weights_only=True,
            map_location='cpu',
            mmap=True,
        )
        cls.original_teacher = torch.load(
            os.environ['TORCHEEG_NEURORVQ_TOKENIZER_CHECKPOINT'],
            weights_only=True,
            map_location='cpu',
            mmap=True,
        )
        cls.values = torch.sin(torch.arange(2400).float() / 31).reshape(2, 3, 400)
        cls.mask = torch.tensor(
            [
                [[True, False], [False, True], [False, False]],
                [[False, False], [True, False], [False, False]],
            ]
        )

    @classmethod
    def tearDownClass(cls):
        # Keep full-suite memory bounded before the separate tokenizer oracle.
        for name in [
            'model',
            'reference_backbone',
            'reference_teacher',
            'original_backbone',
            'original_teacher',
            'values',
            'mask',
        ]:
            delattr(cls, name)

    def setUp(self):
        self.model.backbone.backend.load_state_dict(self.original_backbone, strict=True)
        self.model.tokenizer.backend.load_state_dict(self.original_teacher, strict=True)
        self.reference_backbone.load_state_dict(self.original_backbone, strict=True)
        self.reference_teacher.load_state_dict(self.original_teacher, strict=True)
        self.model.zero_grad(set_to_none=True)
        self.reference_backbone.zero_grad(set_to_none=True)

    def test_full_native_targets_both_logits_all_gradients_and_frozen_teacher(self):
        self.assertEqual(len(self.model.state_dict()), 841)
        self.assertEqual(len(self.model.tokenizer.backend.state_dict()), 505)
        self.assertEqual(len(self.model.backbone.backend.state_dict()), 330)
        temporal = torch.tensor([[254, 255] * 3]).expand(2, -1)
        spatial = torch.tensor([[0, 0, 55, 55, 56, 56]]).expand(2, -1)
        for training in [False, True]:
            self.setUp()
            self.model.train(training)
            self.reference_backbone.train(training)
            self.reference_teacher.eval()
            self.assertFalse(self.model.tokenizer.training)
            teacher_before = digest(self.model.tokenizer.state_dict())
            left = self.values.clone().requires_grad_()
            right = self.values.clone().requires_grad_()
            with torch.no_grad():
                # Literal source token format, independently flattened by scale/residual.
                native_tokens = self.reference_teacher.get_tokens(
                    right.detach().reshape(2, 3, 2, 200), temporal, spatial
                )['token']
                targets = native_tokens.reshape(32, 2, 3, 2)
            original = self.reference_backbone(
                right.reshape(2, 3, 2, 200),
                temporal,
                spatial,
                bool_masked_pos=self.mask.flatten(1),
                use_for_pretraining=True,
            )
            actual = self.model(left, self.mask)
            torch.testing.assert_close(actual['targets'], targets, rtol=0, atol=0)
            for name, logits in zip(['logits', 'symmetric_logits'], original):
                torch.testing.assert_close(actual[name], logits, rtol=0, atol=0)
            # Explicit TorchEEG CE definition, not a nonexistent author trainer.
            selected = targets.flatten(1)[:, self.mask.flatten()]
            complement = targets.flatten(1)[:, ~self.mask.flatten()]
            masked_loss = F.cross_entropy(
                original[0].reshape(-1, 8192), selected.reshape(-1)
            )
            symmetric_loss = F.cross_entropy(
                original[1].reshape(-1, 8192), complement.reshape(-1)
            )
            expected_loss = masked_loss + symmetric_loss
            for name, expected in [
                ('masked_loss', masked_loss),
                ('symmetric_loss', symmetric_loss),
                ('loss', expected_loss),
            ]:
                torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)
            actual['loss'].backward()
            expected_loss.backward()
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
            reference_parameters = dict(self.reference_backbone.named_parameters())
            count = 0
            for name, parameter in self.model.backbone.backend.named_parameters():
                reference = reference_parameters[name]
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                torch.testing.assert_close(
                    parameter.grad, reference.grad, rtol=0, atol=0, msg=name
                )
                count += 1
            self.assertEqual(count, 330)
            for index in range(1, 33):
                head = getattr(self.model.backbone.backend, 'head_pre_' + str(index))
                self.assertGreater(head.weight.grad.abs().sum().item(), 0)
            self.assertTrue(
                all(
                    parameter.grad is None
                    for parameter in self.model.tokenizer.parameters()
                )
            )
            self.assertEqual(digest(self.model.tokenizer.state_dict()), teacher_before)

    def test_complete_optimizer_global_mask_rng_and_teacher_resume(self):
        model = self.model.train()
        optimizer = torch.optim.AdamW(
            model.backbone.parameters(), lr=1e-5, weight_decay=1e-5
        )
        generator = torch.Generator().manual_seed(17)
        values = self.values[:1]
        original_teacher = digest(model.tokenizer.state_dict())

        def step():
            eeg = values + torch.randn_like(values) * 0.01
            mask = self.example.make_mask(eeg, generator)
            optimizer.zero_grad(set_to_none=True)
            result = model(eeg, mask)
            result['loss'].backward()
            optimizer.step()
            return result['loss'].detach().clone()

        torch.manual_seed(42)
        step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'complete-resume.pt'
            torch.save(
                dict(
                    state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    rng=torch.get_rng_state(),
                    mask_rng=generator.get_state(),
                ),
                path,
            )
            expected_loss = step()
            expected_model = digest(model.state_dict())
            expected_optimizer = digest(optimizer.state_dict())
            expected_rng = torch.get_rng_state()
            expected_mask_rng = generator.get_state()
            payload = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
            model.load_state_dict(payload['state_dict'], strict=True)
            optimizer.load_state_dict(payload['optimizer'])
            torch.set_rng_state(payload['rng'])
            generator.set_state(payload['mask_rng'])
            actual_loss = step()
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
            self.assertEqual(digest(model.state_dict()), expected_model)
            self.assertEqual(digest(optimizer.state_dict()), expected_optimizer)
            torch.testing.assert_close(
                torch.get_rng_state(), expected_rng, rtol=0, atol=0
            )
            torch.testing.assert_close(
                generator.get_state(), expected_mask_rng, rtol=0, atol=0
            )
        self.assertEqual(digest(model.tokenizer.state_dict()), original_teacher)


if __name__ == '__main__':
    unittest.main()
