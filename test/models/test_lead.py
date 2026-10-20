import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from safetensors.torch import save_file

from torcheeg.models import LEAD, LEADClassifier


CHANNELS = ['Fp1', 'Cz', 'O2']


class NativeStyleBackend(nn.Module):
    """Independently written protocol fixture, not bundled LEAD author code.

    Tests adapter gradients, physical metadata and geometry persistence. The
    complete licensed native model/checkpoint is validated separately.
    """
    def __init__(self, dataset_id=0, num_channels=3):
        super().__init__()
        self.d_model, self.patch_len, self.stride = 8, 2, 2
        self.enc_embedding = nn.Module()
        self.enc_embedding.coord_buffer_names = {dataset_id: 'coords'}
        self.enc_embedding.register_buffer('coords', torch.arange(num_channels*3).float().reshape(num_channels, 3)/10, persistent=False)
        self.enc_embedding.use_sampling_embedding = True
        self.enc_embedding.sr_embeddings = nn.ParameterDict({
            'hz_'+str(rate): nn.Parameter(torch.full((8,), rate/1000)) for rate in [50, 100, 200]})
        self.projection = nn.Linear(num_channels, 8)
        self.norm = nn.LayerNorm(8)
        self.dropout = nn.Dropout(.2)
        self.last_metadata = None

    def encode(self, x, metadata):
        self.last_metadata = metadata.detach().clone()
        coords = self.enc_embedding.coords
        adjusted = x*(1+coords[:, 0])
        return self.dropout(self.norm(self.projection(adjusted.mean(1)) +
                                      self.enc_embedding.sr_embeddings['hz_200']))


class TestLEAD(unittest.TestCase):
    def test_adapter_matches_backend_outputs_inputs_and_all_parameter_gradients(self):
        native = NativeStyleBackend()
        wrapped = LEAD(copy.deepcopy(native), CHANNELS)
        for mode in [False, True]:
            native.train(mode); wrapped.train(mode)
            native.zero_grad(set_to_none=True); wrapped.zero_grad(set_to_none=True)
            x = torch.sin(torch.arange(120).float()/17).reshape(2, 3, 20).requires_grad_()
            xx = x.detach().clone().requires_grad_()
            torch.manual_seed(71); expected = native.encode(x.transpose(1, 2), torch.tensor([[0, 0, 200, 0]]).expand(2, -1))
            torch.manual_seed(71); actual = wrapped(xx)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            expected.square().mean().backward(); actual.square().mean().backward()
            torch.testing.assert_close(x.grad, xx.grad, rtol=0, atol=0)
            for name, p in native.named_parameters():
                actual_grad = dict(wrapped.backend.named_parameters())[name].grad
                if p.grad is None: self.assertIsNone(actual_grad)
                else: torch.testing.assert_close(actual_grad, p.grad, rtol=0, atol=0)
            self.assertTrue((wrapped.backend.last_metadata[:, :2] == 0).all())
            self.assertTrue((wrapped.backend.last_metadata[:, 2] == 200).all())

    def test_geometry_persistence_restore_and_dtype(self):
        model = LEAD(NativeStyleBackend(), CHANNELS).eval()
        eeg = torch.randn(2, 3, 20)
        expected = model(eeg)
        state = {name: value.clone() for name, value in model.state_dict().items()}
        self.assertIn('electrode_coords', state)
        self.assertNotIn('backend.enc_embedding.coords', state)
        backend = NativeStyleBackend()
        backend.enc_embedding.coords.add_(100)
        restored = LEAD(backend, CHANNELS).eval()
        restored.load_state_dict(state, strict=True)
        torch.testing.assert_close(restored(eeg), expected, rtol=0, atol=0)
        self.assertIs(restored.backend.enc_embedding.coords, restored.electrode_coords)
        restored.double()
        self.assertEqual(restored(eeg.double()).dtype, torch.float64)
        self.assertIs(restored.backend.enc_embedding.coords, restored.electrode_coords)
        torch.testing.assert_close(restored.electrode_coords, state['electrode_coords'].double(), rtol=0, atol=0)
        for changes in [dict(channel_names=CHANNELS[::-1]), dict(sampling_rate=100), dict(dataset_id=1)]:
            dataset_id = changes.get('dataset_id', 0)
            other = LEAD(NativeStyleBackend(dataset_id), **dict(dict(channel_names=CHANNELS), **changes))
            with self.assertRaises(RuntimeError): other.load_state_dict(state, strict=True)
        bad = dict(state, electrode_coords=torch.full_like(state['electrode_coords'], float('nan')))
        with self.assertRaises(RuntimeError): model.load_state_dict(bad, strict=True)

    @unittest.skipUnless(hasattr(nn.Module, 'to_empty'), 'Module.to_empty is unavailable.')
    def test_to_empty_then_load_preserves_configuration_geometry_and_predictions(self):
        model = LEAD(NativeStyleBackend(), CHANNELS).eval()
        eeg = torch.randn(2, 3, 20)
        expected = model(eeg)
        state = {name: value.clone() for name, value in model.state_dict().items()}
        restored = LEAD(NativeStyleBackend(), CHANNELS).eval().to_empty(device='cpu')
        restored.load_state_dict(state, strict=True)
        torch.testing.assert_close(restored(eeg), expected, rtol=0, atol=0)
        self.assertIs(restored.backend.enc_embedding.coords, restored.electrode_coords)

    def test_batch_isolation_singletons_and_geometry_changes(self):
        model = LEAD(NativeStyleBackend(), CHANNELS).eval()
        x = torch.randn(2, 3, 20, requires_grad=True)
        output = model(x)
        output[0].sum().backward()
        self.assertEqual(x.grad[1].count_nonzero().item(), 0)
        torch.testing.assert_close(model(x[:1]), output[:1], rtol=1e-5, atol=1e-6)
        self.assertEqual(model(x[:1, :, :1]).shape, (1, 8))
        with torch.no_grad(): model.electrode_coords[0, 0].add_(1)
        self.assertFalse(torch.equal(model(x), output))

    def test_classifier_optimizer_and_frozen_state(self):
        model = LEADClassifier(LEAD(NativeStyleBackend(), CHANNELS), freeze_backbone=True).train()
        state = {name: value.clone() for name, value in model.backbone.state_dict().items()}
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.classifier.training)
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=.01)
        head = model.classifier.weight.detach().clone()
        loss = nn.functional.cross_entropy(model(torch.randn(2, 3, 20)), torch.tensor([0, 1]))
        loss.backward(); optimizer.step()
        self.assertFalse(torch.equal(head, model.classifier.weight))
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        for name, value in state.items():
            torch.testing.assert_close(model.backbone.state_dict()[name], value, rtol=0, atol=0)
        unfrozen = LEADClassifier(LEAD(NativeStyleBackend(), CHANNELS))
        unfrozen(torch.randn(2, 3, 20)).square().sum().backward()
        self.assertIsNotNone(unfrozen.backbone.backend.projection.weight.grad)

    def test_complete_safe_checkpoint_loading_and_failures(self):
        state = NativeStyleBackend().state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            torch.save(state, path)
            model = LEAD.from_pretrained(path, NativeStyleBackend(), CHANNELS)
            for key, value in state.items():
                torch.testing.assert_close(model.backend.state_dict()[key], value, rtol=0, atol=0)
            torch.save({'module.'+key: value for key, value in state.items()}, path)
            LEAD.from_pretrained(path, NativeStyleBackend(), CHANNELS, prefix='module.')
            with self.assertRaises(ValueError): LEAD.from_pretrained(path, NativeStyleBackend(), CHANNELS)
            key = 'projection.weight'
            for value in [None, state[key][:1], state[key].double(), torch.full_like(state[key], float('nan'))]:
                bad = dict(state)
                if value is None: del bad[key]
                else: bad[key] = value
                torch.save(bad, path)
                with self.assertRaises(ValueError): LEAD.from_pretrained(path, NativeStyleBackend(), CHANNELS)
            for bad in [dict(state, extra=torch.zeros(1)), dict(model=state), dict(state, wrong='object')]:
                torch.save(bad, path)
                with self.assertRaises(ValueError): LEAD.from_pretrained(path, NativeStyleBackend(), CHANNELS)
            safe = Path(directory)/'model.safetensors'; save_file(state, safe)
            LEAD.from_pretrained(safe, NativeStyleBackend(), CHANNELS)

    def test_invalid_metadata_inputs_and_protocol(self):
        for names in [[], ['Cz', 'Cz'], ['Cz', None, 'O2'], ['Cz']]:
            with self.assertRaises(ValueError): LEAD(NativeStyleBackend(), names)
        for kwargs in [dict(sampling_rate=0), dict(sampling_rate=512), dict(dataset_id=-1), dict(dataset_id=2)]:
            with self.assertRaises(ValueError): LEAD(NativeStyleBackend(), CHANNELS, **kwargs)
        with self.assertRaises(ValueError): LEAD(nn.Linear(3, 8), CHANNELS)
        backend = NativeStyleBackend(); backend.enc_embedding.coords[0, 0] = float('nan')
        with self.assertRaises(ValueError): LEAD(backend, CHANNELS)
        model = LEAD(NativeStyleBackend(), CHANNELS)
        for eeg in [torch.zeros(2, 3, 20).long(), torch.zeros(2, 2, 20), torch.zeros(0, 3, 20), torch.zeros(1, 3, 0)]:
            with self.assertRaises(ValueError): model(eeg)
        with self.assertRaises(ValueError): LEADClassifier(model, freeze_backbone=1)
        with self.assertRaises(ValueError): LEADClassifier(model, num_classes=0)
        with self.assertRaises(ValueError): LEAD.from_pretrained('missing', model.backend, CHANNELS, prefix=1)
        model.backend.encode = lambda x, metadata: torch.zeros(x.shape[0], 2)
        with self.assertRaises(ValueError): model(torch.zeros(1, 3, 20))


if __name__ == '__main__':
    unittest.main()
