"""Independent protocol fixtures; the optional native oracle checks author models."""

import copy
import tempfile
from pathlib import Path
import unittest

import torch
from torch import nn

from torcheeg.models import SleepFM, SleepFMStaging, SleepFMDiagnosis, SleepFMClassifier


class UpstreamFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(640, 128)

    def forward(self, x, mask):
        values = self.projection(x.unfold(-1, 640, 640))
        weights = (~mask).to(x.dtype).unsqueeze(-1).unsqueeze(-1)
        tokens = (values * weights).sum(1) / weights.sum(1)
        return tokens.mean(1), tokens


class StagingFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 5)

    def forward(self, x, mask):
        values = x.mean(1)
        return self.fc(self.lstm(values)[0]), mask[:, 0]


class DiagnosisFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(130, 1065)

    def forward(self, x, mask, demographics):
        weights = (~mask).to(x.dtype).unsqueeze(-1)
        features = (x * weights).sum((1, 2)) / weights.sum((1, 2))
        return self.classifier(torch.cat((features, demographics), -1))


class TestSleepFM(unittest.TestCase):
    def test_upstream_features_mask_gradient(self):
        torch.manual_seed(2)
        model = SleepFM(UpstreamFixture(), ['Fp1', 'Fp2'])
        x = torch.randn(2, 2, 1280, requires_grad=True)
        mask = torch.tensor([[False, False], [False, True]])
        pooled, tokens = model.backend(x, mask)
        torch.testing.assert_close(model(x, mask), pooled, rtol=0, atol=0)
        torch.testing.assert_close(
            model(x, mask, return_tokens=True), tokens, rtol=0, atol=0
        )
        model(x, mask).square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(torch.count_nonzero(x.grad[1, 1]), 0)
        changed = x.detach().clone()
        changed[1, 1] += 100
        torch.testing.assert_close(model(changed, mask), pooled, rtol=0, atol=0)

    def test_staging_padding_predictions_and_gradients(self):
        torch.manual_seed(3)
        model = SleepFMStaging(StagingFixture(), ['BAS', 'RESP'])
        x = torch.randn(2, 2, 5, 128, requires_grad=True)
        mask = torch.zeros(2, 2, 5, dtype=torch.bool)
        mask[1, :, 3:] = True
        logits, temporal = model(x, mask)
        self.assertEqual(logits.shape, (2, 5, 5))
        self.assertTrue(torch.equal(temporal, mask[:, 0]))
        expected = model.backend(x[1:2, :, :3], mask[1:2, :, :3])[0]
        torch.testing.assert_close(logits[1:2, :3], expected, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(logits[1, 3:]), 0)
        changed = x.detach().clone()
        changed[1, :, 3:] += 100
        torch.testing.assert_close(model(changed, mask)[0], logits, rtol=0, atol=0)
        logits.square().sum().backward()
        self.assertEqual(torch.count_nonzero(x.grad[1, :, 3:]), 0)
        self.assertTrue(torch.isfinite(x.grad).all())
        # The independent bidirectional LSTM fixture confirms that passing
        # padded future input into the native model would change valid logits.
        naive = model.backend(x, mask)[0]
        self.assertFalse(torch.equal(naive[1, :3], logits[1, :3]))

    def test_diagnosis_all_columns_demographics_and_optimization(self):
        model = SleepFMDiagnosis(
            DiagnosisFixture(), ['BAS', 'RESP'], ['encoded_age', 'encoded_gender']
        )
        x = torch.randn(2, 2, 4, 128, requires_grad=True)
        demo = torch.randn(2, 2, requires_grad=True)
        mask = torch.zeros(2, 2, 4, dtype=torch.bool)
        mask[1, :, -1] = True
        mask[1, 1] = True
        actual = model(x, demo, mask)
        expected = model.backend(x, mask, demo)
        self.assertEqual(actual.shape, (2, 1065))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.backend.classifier.weight.detach().clone()
        actual.square().mean().backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(demo.grad).all())
        self.assertFalse(torch.equal(before, model.backend.classifier.weight))
        self.assertEqual(torch.count_nonzero(x.grad[1, 1]), 0)
        self.assertEqual(torch.count_nonzero(x.grad[1, :, -1]), 0)
        for bad in [
            torch.zeros(2, 1),
            torch.full((2, 2), float('nan')),
            torch.zeros(2, 2, dtype=torch.long),
        ]:
            with self.assertRaises(ValueError):
                model(x, bad, mask)

    def test_complete_loader_and_metadata_restore(self):
        backend = UpstreamFixture()
        state = backend.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(
                {'state_dict': {'module.' + k: v for k, v in state.items()}}, path
            )
            model = SleepFM.from_pretrained(path, UpstreamFixture(), ['Fp1', 'Fp2'])
            x = torch.randn(2, 2, 640)
            saved = copy.deepcopy(model.state_dict())
            restored = SleepFM(UpstreamFixture(), ['Fp1', 'Fp2'])
            restored.to_empty(device='cpu')
            restored.load_state_dict(saved)
            torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
            with self.assertRaises(RuntimeError):
                SleepFM(UpstreamFixture(), ['Fp2', 'Fp1']).load_state_dict(saved)
            with self.assertRaises(RuntimeError):
                SleepFM(
                    UpstreamFixture(), ['Fp1', 'Fp2'], modality='EMG'
                ).load_state_dict(saved)
            for bad in [
                dict(state, extra=torch.zeros(1)),
                dict(list(state.items())[1:]),
                dict(
                    state, **{'projection.weight': state['projection.weight'].double()}
                ),
                dict(
                    state,
                    **{
                        'projection.weight': torch.full_like(
                            state['projection.weight'], float('inf')
                        )
                    },
                ),
            ]:
                torch.save(bad, path)
                with self.assertRaises(ValueError):
                    SleepFM.from_pretrained(
                        path,
                        UpstreamFixture(),
                        ['Fp1', 'Fp2'],
                        prefix='',
                        state_key=None,
                    )
            from safetensors.torch import save_file

            path = Path(directory) / 'model.safetensors'
            save_file(state, str(path))
            SleepFM.from_pretrained(
                path, UpstreamFixture(), ['Fp1', 'Fp2'], prefix='', state_key=None
            )

    def test_new_classifier_freezes_native_state(self):
        model = SleepFMClassifier(
            SleepFM(UpstreamFixture(), ['Fp1', 'Fp2']), 3, freeze_backbone=True
        ).train()
        self.assertFalse(model.backbone.training)
        optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.1)
        before = model.classifier.weight.detach().clone()
        nn.functional.cross_entropy(
            model(torch.randn(2, 2, 640)), torch.tensor([0, 2])
        ).backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.classifier.weight))
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

    def test_invalid_input_and_unsupported_masks(self):
        base = SleepFM(UpstreamFixture(), ['Fp1', 'Fp2'])
        for x in [
            torch.randn(2, 1, 640),
            torch.randn(2, 2, 641),
            torch.randn(0, 2, 640),
            torch.randn(1, 2, 82560),
        ]:
            with self.assertRaises(ValueError):
                base(x)
        with self.assertRaises(ValueError):
            base(torch.randn(2, 2, 640), torch.ones(2, 2, dtype=torch.bool))
        model = SleepFMStaging(StagingFixture(), ['BAS', 'RESP'])
        x = torch.randn(2, 2, 4, 128)
        bads = []
        mask = torch.zeros(2, 2, 4, dtype=torch.bool)
        mask[1, 1, 2] = True
        bads.append(mask)
        mask = torch.zeros(2, 2, 4, dtype=torch.bool)
        mask[1, :, 1] = True
        bads.append(mask)
        mask = torch.zeros(2, 2, 4, dtype=torch.bool)
        mask[1, 0] = True
        bads.append(mask)
        for mask in bads:
            with self.assertRaises(ValueError):
                model(x, mask)
        with self.assertRaises(ValueError):
            SleepFMDiagnosis(DiagnosisFixture(), ['BAS'], ['age'])
        with self.assertRaises(ValueError):
            SleepFMStaging(StagingFixture(), ['UNKNOWN'])
        with self.assertRaises(ValueError):
            SleepFMStaging(StagingFixture(), ['BAS', 'BAS'])


if __name__ == '__main__':
    unittest.main()
