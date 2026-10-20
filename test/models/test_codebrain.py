import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from torcheeg.models import CodeBrain, CodeBrainClassifier


SMALL = dict(hid_channels=8, depth=2, max_length=16, num_time_tokens=7, num_frequency_tokens=9)


class TinyCodeBrain(CodeBrain):
    def __init__(self, **kwargs):
        super().__init__(**dict(SMALL, **kwargs))


class TestCodeBrain(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.eeg = torch.randn(2, 3, 2, 200)

    def test_official_reference_features_tokens_and_every_gradient(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/codebrain_reference.json').read_text())
        archive = np.load(io.BytesIO(base64.b64decode(fixture['state_npz_base64'])), allow_pickle=False)
        state = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
        archive.close()
        model = CodeBrain(**fixture['config']).eval()
        model.load_state_dict(state, strict=True)
        for name in ['eval', 'masked']:
            case = fixture['cases'][name]
            x = torch.linspace(-1, 1, 2*3*2*200).reshape(2, 3, 2, 200).requires_grad_()
            mask = torch.tensor(case['mask']) if case['mask'] is not None else None
            model.zero_grad(set_to_none=True)
            output = model(x, mask)
            torch.testing.assert_close(output, torch.tensor(case['output']), rtol=3e-5, atol=3e-6)
            if mask is None:
                coefficients = torch.sin(torch.arange(output.numel()).reshape_as(output)/13)
                loss = (output*coefficients).square().mean()
            else:
                t, f = model.lm_head_t(output[mask]), model.lm_head_f(output[mask])
                loss = t.square().mean()+f.square().mean()
            loss.backward()
            torch.testing.assert_close(x.grad, torch.tensor(case['eeg_gradient']), rtol=5e-4, atol=3e-6)
            for parameter_name, parameter in model.named_parameters():
                with self.subTest(case=name, parameter=parameter_name):
                    expected = case['parameter_gradients'][parameter_name]
                    if expected is None:
                        self.assertIsNone(parameter.grad)
                        continue
                    values = parameter.grad.double().flatten()
                    torch.testing.assert_close(values[:8], torch.tensor(expected['first'], dtype=torch.float64), rtol=5e-4, atol=3e-6)
                    self.assertAlmostEqual(values.sum().item(), expected['sum'], delta=3e-5)
                    self.assertAlmostEqual(values.norm().item(), expected['norm'], delta=3e-5)

    def test_batch_and_singleton_axes_and_isolation(self):
        model = TinyCodeBrain().eval()
        model(self.eeg)  # initialize source-independent kernel buffers
        x = self.eeg.clone().requires_grad_()
        output = model(x)
        torch.testing.assert_close(output, torch.cat([model(x[i:i+1]) for i in range(2)]), rtol=3e-5, atol=3e-6)
        output[0].square().sum().backward()
        torch.testing.assert_close(x.grad[1], torch.zeros_like(x.grad[1]))
        self.assertEqual(model(torch.randn(1, 1, 1, 200)).shape, (1, 1, 1, 200))

    def test_mask_removes_raw_input_and_predicts_selected_tokens(self):
        model = TinyCodeBrain().eval()
        mask = torch.zeros(2, 3, 2, dtype=torch.bool)
        mask[:, 0, 0] = True
        x = self.eeg.clone().requires_grad_()
        t, f = model.predict_tokens(x, mask)
        self.assertEqual(t.shape, (2, 7)); self.assertEqual(f.shape, (2, 9))
        (t.square().mean()+f.square().mean()).backward()
        torch.testing.assert_close(x.grad[mask], torch.zeros_like(x.grad[mask]))
        changed = self.eeg.clone(); changed[mask] += 100
        actual = model.predict_tokens(changed, mask)
        torch.testing.assert_close(actual[0], t); torch.testing.assert_close(actual[1], f)
        for p in [model.lm_head_t.weight, model.lm_head_f.weight]:
            self.assertGreater(p.grad.abs().sum().item(), 0)
        empty = model.predict_tokens(self.eeg, torch.zeros_like(mask))
        self.assertEqual(empty[0].shape, (0, 7))

    def test_kernel_initialization_restore_and_input_independence(self):
        model = TinyCodeBrain().eval()
        other = TinyCodeBrain().eval()
        other.load_state_dict(model.state_dict(), strict=True)
        before = copy.deepcopy(model.state_dict())
        model(self.eeg)
        other(torch.randn(1, 5, 5, 200))
        for name, value in model.state_dict().items():
            if name.endswith('kernel_norm') or name.endswith('kernel_norm_initialized'):
                torch.testing.assert_close(value, other.state_dict()[name], rtol=0, atol=0)
            else:
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        restored = TinyCodeBrain().eval(); restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(self.eeg), model(self.eeg), rtol=0, atol=0)
        # Loading an uninitialized state must reset the derived Python flag.
        restored.load_state_dict(before, strict=True)
        self.assertFalse(restored.residual_layer.residual_blocks[0].S41._kernel_norm_ready)
        torch.testing.assert_close(restored(self.eeg), model(self.eeg), rtol=0, atol=0)

    def test_training_dropout_seed_and_optimizer(self):
        model = TinyCodeBrain().train()
        torch.manual_seed(7); first = model(self.eeg)
        torch.manual_seed(7); second = model(self.eeg)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.manual_seed(8); self.assertFalse(torch.equal(first, model(self.eeg)))
        before = model.final_conv[2].conv.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        weights = torch.linspace(-1, 1, first.numel()).reshape_as(first)
        (first*weights).square().mean().backward(); optimizer.step()
        self.assertFalse(torch.equal(before, model.final_conv[2].conv.weight))

    def test_classifier_freeze_safe_export_and_last_batch_one(self):
        model = CodeBrainClassifier(TinyCodeBrain(), 3, 2, 2, dropout=0, freeze_backbone=True)
        model.train(); self.assertFalse(model.backbone.training)
        before = {name: p.detach().clone() for name, p in model.backbone.named_parameters()}
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=.01)
        torch.nn.functional.cross_entropy(model(self.eeg), torch.tensor([0, 1])).backward(); optimizer.step()
        self.assertEqual(model(self.eeg[:1]).shape, (1, 2))
        for name, parameter in model.backbone.named_parameters():
            self.assertIsNone(parameter.grad); torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model.pt'; torch.save(model.state_dict(), path)
            other = CodeBrainClassifier(TinyCodeBrain(), 3, 2, 2, dropout=0, freeze_backbone=True)
            other.load_state_dict(torch.load(path, weights_only=True), strict=True)
            torch.testing.assert_close(model(self.eeg), other(self.eeg), rtol=0, atol=0)

    def test_safe_loader_strict_and_corrupt_states(self):
        state = TinyCodeBrain().state_dict()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model.pt'
            torch.save({'module.'+k: v for k,v in state.items()}, path)
            loaded = TinyCodeBrain.from_pretrained(path)
            for name,value in state.items(): torch.testing.assert_close(loaded.state_dict()[name],value,rtol=0,atol=0)
            torch.save(state,path); TinyCodeBrain.from_pretrained(path,prefix='')
            key='lm_head_t.weight'
            for replacement in [None,state[key][:1],state[key].double(),torch.full_like(state[key],float('inf'))]:
                bad=dict(state)
                if replacement is None: del bad[key]
                else: bad[key]=replacement
                torch.save(bad,path)
                with self.assertRaises(ValueError): TinyCodeBrain.from_pretrained(path,prefix='')
            for bad in [dict(state, unexpected=torch.zeros(1)),{'model':state},[torch.zeros(1)],
                        dict(state, **{'residual_layer.residual_blocks.0.S41.kernel_norm':torch.zeros(8,16,1)})]:
                torch.save(bad,path)
                with self.assertRaises(ValueError): TinyCodeBrain.from_pretrained(path,prefix='')

    def test_invalid_configuration_and_input(self):
        for kwargs in [dict(hid_channels=3),dict(depth=0),dict(dropout=1),dict(max_length=1),dict(num_time_tokens=True)]:
            with self.assertRaises(ValueError): TinyCodeBrain(**kwargs)
        model=TinyCodeBrain()
        for x,mask in [(self.eeg.long(),None),(self.eeg[..., :199],None),(self.eeg[:0],None),
                       (self.eeg,torch.zeros(2,3,2)),(self.eeg,torch.ones(2,3,3,dtype=torch.bool))]:
            with self.assertRaises(ValueError): model(x,mask)
        with self.assertRaises(ValueError): model.predict_tokens(self.eeg,None)
        with self.assertRaises(ValueError): TinyCodeBrain.from_pretrained('missing',prefix=None)
        with self.assertRaises(ValueError): CodeBrainClassifier(model,num_classes=0)
        with self.assertRaises(ValueError): CodeBrainClassifier(model,num_electrodes=19)(self.eeg)


if __name__ == '__main__':
    unittest.main()
