"""Independent interface fixtures, distinct from the full author-model oracle."""
import copy
import tempfile
from pathlib import Path
import unittest

import torch
from torch import nn

from torcheeg.models import NeuroRVQ, NeuroRVQClassifier


class InterfaceBackend(nn.Module):
    """Small independently authored protocol fixture; not a NeuroRVQ replica."""
    def __init__(self, vocab_size=3):
        super().__init__()
        self.patch_size = self.embed_dim = 200
        self.use_as_encoder = self.use_for_pretraining = True
        self.pos_embed = nn.Parameter(torch.zeros(vocab_size+1, 200))
        self.time_embed = nn.Parameter(torch.zeros(256, 200))
        self.projection = nn.Linear(200, 200)
        # Shared heads reduce fixture memory; only the native oracle checks
        # the 32 separately trained released heads and complete architecture.
        head = nn.Linear(200, 8192)
        for index in range(1, 33):
            setattr(self, 'head_pre_'+str(index), head)

    def forward(self, x, temporal_embedding_ix, spatial_embedding_ix,
                return_patch_tokens=False, bool_masked_pos=None, use_for_pretraining=False):
        x = self.projection(x.flatten(1, 2))
        x = x + self.time_embed[temporal_embedding_ix] + self.pos_embed[spatial_embedding_ix]
        if bool_masked_pos is not None:
            logits = self.head_pre_1(x)
            return (torch.stack([logits[bool_masked_pos]]*32),
                    torch.stack([logits[~bool_masked_pos]]*32))
        return x, x*2, x*3, x*4, 200


class TestNeuroRVQ(unittest.TestCase):
    def make(self, names=('Fp1', 'Fp2')):
        return NeuroRVQ(InterfaceBackend(), list(names), ['fp1', 'fp2', 'cz'])

    def test_features_indices_gradient_and_tokens(self):
        torch.manual_seed(3)
        model = self.make().double()
        x = torch.randn(2, 2, 400, dtype=torch.float64, requires_grad=True)
        patches = x.reshape(2, 2, 2, 200)
        branches = model.backend(patches, torch.tensor([[254, 255, 254, 255]]).expand(2, -1),
                                 torch.tensor([[0, 0, 1, 1]]).expand(2, -1), return_patch_tokens=True)
        expected = torch.cat(branches[:4], -1).reshape(2, 2, 2, 800)
        actual = model(x, return_tokens=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(model(x), expected.flatten(1, 2).mean(1), rtol=0, atol=0)
        ga = torch.autograd.grad(actual.square().sum(), (x, model.backend.projection.weight), retain_graph=True)
        gb = torch.autograd.grad(expected.square().sum(), (x, model.backend.projection.weight))
        for left, right in zip(ga, gb):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        changed = x.detach().clone(); changed[1] += 10
        torch.testing.assert_close(model(changed)[0], model(x)[0], rtol=0, atol=0)

    def test_masks_and_optimization(self):
        model = self.make()
        x = torch.randn(2, 2, 400, requires_grad=True)
        mask = torch.tensor([[[True, False], [False, True]], [[True, True], [False, False]]])
        a, b = model.forward_masked(x, mask)
        self.assertEqual(a.shape, (32, 4, 8192)); self.assertEqual(b.shape, (32, 4, 8192))
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        before = model.backend.projection.weight.detach().clone()
        loss = a.square().mean()+b.square().mean()
        loss.backward(); optimizer.step()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertFalse(torch.equal(before, model.backend.projection.weight))
        for bad in [mask.long(), mask[:, :1], torch.ones_like(mask), torch.zeros_like(mask)]:
            with self.assertRaises(ValueError): model.forward_masked(x, bad)

    def test_strict_restore_and_metadata(self):
        model = self.make().eval(); x = torch.randn(2, 2, 400)
        state = copy.deepcopy(model.state_dict())
        restored = self.make().eval()
        restored.to_empty(device='cpu')
        restored.load_state_dict(state, strict=True)
        torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
        with self.assertRaises(RuntimeError): self.make(('Fp2', 'Fp1')).load_state_dict(state)
        bad = dict(state); bad['spatial_ids'] = state['spatial_ids'].float()
        with self.assertRaises(RuntimeError): restored.load_state_dict(bad)

    def test_safe_complete_checkpoint(self):
        backend = InterfaceBackend()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'weights.pt'
            state = backend.state_dict()
            torch.save(state, path)
            loaded = NeuroRVQ.from_pretrained(path, InterfaceBackend(), ['fp1', 'fp2'], ['fp1', 'fp2', 'cz'])
            for name, value in loaded.backend.state_dict().items():
                torch.testing.assert_close(value, state[name], rtol=0, atol=0)
            for bad in [dict(list(state.items())[1:]), dict(state, extra=torch.zeros(1)),
                        dict(state, pos_embed=state['pos_embed'].double()),
                        dict(state, pos_embed=torch.full_like(state['pos_embed'], float('nan')))]:
                torch.save(bad, path)
                with self.assertRaises(ValueError):
                    NeuroRVQ.from_pretrained(path, InterfaceBackend(), ['fp1', 'fp2'], ['fp1', 'fp2', 'cz'])
            torch.save({'native.'+k:v for k,v in state.items()}, path)
            NeuroRVQ.from_pretrained(path, InterfaceBackend(), ['fp1', 'fp2'], ['fp1', 'fp2', 'cz'], prefix='native.')
            from safetensors.torch import save_file
            # Fixture heads share storage: explicitly serialize independent views.
            path = Path(directory)/'weights.safetensors'
            save_file({k:v.clone() for k,v in state.items()}, str(path))
            NeuroRVQ.from_pretrained(path, InterfaceBackend(), ['fp1', 'fp2'], ['fp1', 'fp2', 'cz'])

    def test_classifier_freezing_and_training(self):
        model = NeuroRVQClassifier(self.make(), 3, freeze_backbone=True).train()
        self.assertFalse(model.backbone.training)
        x = torch.randn(2, 2, 400)
        before = model.classifier.weight.detach().clone()
        optimizer = torch.optim.SGD(model.classifier.parameters(), lr=.1)
        nn.functional.cross_entropy(model(x), torch.tensor([0, 2])).backward(); optimizer.step()
        self.assertFalse(torch.equal(before, model.classifier.weight))
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        model = NeuroRVQClassifier(self.make(), freeze_backbone=False)
        model(x).sum().backward()
        self.assertIsNotNone(model.backbone.backend.projection.weight.grad)

    def test_invalid_inputs_and_backend(self):
        model = self.make()
        for value in [torch.ones(2, 2, 400, dtype=torch.long), torch.randn(0, 2, 400),
                      torch.randn(2, 1, 400), torch.randn(2, 2, 399), torch.randn(1, 2, 51400)]:
            with self.assertRaises(ValueError): model(value)
        with self.assertRaises(ValueError): self.make(('fp1', 'FP1'))
        with self.assertRaises(ValueError): self.make(('unknown',))
        with self.assertRaises(ValueError): NeuroRVQ(InterfaceBackend(), ['fp1'], ['fp1','fp2','cz'], sampling_rate=100)
        backend = InterfaceBackend(); del backend.head_pre_32
        with self.assertRaises(ValueError): NeuroRVQ(backend, ['fp1'], ['fp1','fp2','cz'])
        with self.assertRaises(ValueError): model(torch.randn(1, 2, 400), return_tokens=1)
        self.assertEqual(model(torch.randn(1, 2, 200)).shape, (1, 800))


if __name__ == '__main__':
    unittest.main()
