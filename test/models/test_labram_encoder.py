import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from torcheeg.models import LaBraMEncoder


CONFIG = dict(patch_size=16, hid_channels=16, conv_channels=8, mlp_channels=32, depth=2, num_heads=4)


def tensor(record):
    return torch.tensor(record['values'], dtype=torch.float64).reshape(record['shape'])


class TestLaBraMEncoder(unittest.TestCase):
    def test_independent_native_outputs_and_all_input_parameter_gradients(self):
        reference = json.loads((Path(__file__).parent / 'fixtures' / 'labram_reference.json').read_text())
        for case in reference['cases']:
            for entry in case['outputs']:
                with self.subTest(qkv_bias=case['qkv_bias'], qk_norm=case['qk_norm'], mode=entry['mode']):
                    model = LaBraMEncoder(**CONFIG, qkv_bias=case['qkv_bias'], qk_norm=case['qk_norm']).double().eval()
                    model.load_state_dict({k: tensor(v) for k, v in case['state_dict'].items()}, strict=True)
                    x = tensor(entry['input']).requires_grad_()
                    indices = torch.tensor(entry['indices'])
                    if entry['mode'] == 'pooled':
                        actual = model(x, indices)
                    else:
                        actual = model.forward_all(x, indices, normalize=entry['mode'] == 'all')
                    torch.testing.assert_close(actual, tensor(entry['output']), rtol=1e-10, atol=1e-11)
                    coefficient = torch.linspace(-.7, 1.1, actual.numel(), dtype=actual.dtype).reshape_as(actual)
                    (actual * coefficient).sum().backward()
                    torch.testing.assert_close(x.grad, tensor(entry['input_gradient']), rtol=1e-9, atol=1e-10)
                    self.assertEqual(set(dict(model.named_parameters())), set(entry['gradients']))
                    for name, parameter in model.named_parameters():
                        expected = entry['gradients'][name]
                        if expected is None:
                            self.assertIsNone(parameter.grad, name)
                        else:
                            torch.testing.assert_close(parameter.grad, tensor(expected), rtol=1e-9, atol=1e-10, msg=name)

    def test_patch_layout_pooling_and_explicit_signal_identity(self):
        model = LaBraMEncoder(**CONFIG).eval()
        x = torch.randn(2, 2, 48)
        indices = model.channel_indices(['Fp1', 'C3'])
        all_tokens = model.forward_all(x, indices)
        torch.testing.assert_close(all_tokens, model.forward_all(x.reshape(2, 2, 3, 16), indices))
        features = model.forward_features(x, indices)
        self.assertEqual(features.shape, (2, 2, 3, 16))
        torch.testing.assert_close(features.flatten(1, 2), all_tokens[:, 1:])
        raw = model.forward_all(x, indices, normalize=False)
        torch.testing.assert_close(model(x, indices), model.norm(raw[:, 1:].mean(1)))
        self.assertGreater(float((model(x, indices) - features.mean((1, 2))).abs().max()), 1e-5)
        model.pooling = 'cls'
        torch.testing.assert_close(model(x, indices), all_tokens[:, 0])
        batched = torch.stack((indices, indices.flip(0)))
        together = model(x, batched)
        torch.testing.assert_close(together[0], model(x[:1], indices)[0])
        torch.testing.assert_close(together[1], model(x[1:], indices.flip(0))[0])

    def test_channel_catalog_preserves_native_slots_and_rejects_unrepresented_names(self):
        torch.testing.assert_close(LaBraMEncoder.channel_indices([' fp1 ', 'C3', 'P8-O2']), torch.tensor([1, 40, 128]))
        self.assertNotEqual(LaBraMEncoder.channel_indices(['FP1-F7']).item(), LaBraMEncoder.channel_indices(['FP1']).item())
        for names in [[], ['unknown'], ['C3-REF'], ['C3', ' c3 '], ['FP1-F3'], ['P4-O2'], 'C3', [None]]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                LaBraMEncoder.channel_indices(names)

    def test_explicit_temporal_interpolation_preserves_original_table(self):
        model = LaBraMEncoder(**CONFIG, max_patches=2, interpolate_time=True).eval()
        x = torch.randn(1, 2, 48)
        captured = {}
        hook = model.pos_drop.register_forward_pre_hook(lambda module, args: captured.update(tokens=args[0]))
        before = model.time_embed.clone()
        model(x, torch.tensor([1, 40]))
        hook.remove()
        temporal = F.interpolate(model.time_embed.transpose(1, 2), size=3, mode='linear', align_corners=False).transpose(1, 2)
        patches = model.patch_embed(x.reshape(1, 2, 3, 16)).reshape(1, 2, 3, 16)
        expected = patches + temporal.unsqueeze(1) + model.pos_embed[:, [1, 40]].unsqueeze(2)
        torch.testing.assert_close(captured['tokens'][:, 1:], expected.flatten(1, 2))
        torch.testing.assert_close(model.time_embed, before, rtol=0, atol=0)
        self.assertEqual(model.time_embed.shape, (1, 2, 16))

    def test_configurations_reject_silent_width_or_behavior_changes(self):
        for kwargs in [dict(patch_size=17), dict(conv_channels=6), dict(hid_channels=17), dict(depth=0),
                       dict(num_heads=3), dict(max_patches=True), dict(qk_norm=1), dict(qkv_bias=1),
                       dict(dropout=1.), dict(attention_dropout=-.1), dict(drop_path_rate=float('nan')),
                       dict(layer_scale=-1.), dict(norm_eps=0.), dict(pooling='typo'), dict(interpolate_time=1)]:
            settings = dict(CONFIG, **kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LaBraMEncoder(**settings)
        model = LaBraMEncoder(**CONFIG, layer_scale=0., use_abs_pos_emb=False, qk_norm=False, qkv_bias=True)
        self.assertIsNone(model.pos_embed)
        self.assertIsNone(model.blocks[0].gamma_1)
        output = model(torch.randn(2, 2, 32), torch.tensor([1, 40]))
        self.assertTrue(torch.isfinite(output).all())

    def test_bad_inputs_and_channel_indices_fail_explicitly(self):
        model = LaBraMEncoder(**CONFIG)
        for x in [torch.ones(2, 2, 32, dtype=torch.long), torch.ones(2, 2, 32, dtype=torch.complex64),
                  torch.randn(0, 2, 32), torch.randn(2, 0, 32), torch.randn(2, 2, 0), torch.randn(2, 2, 33),
                  torch.randn(2, 2, 0, 16), torch.randn(2, 2, 2, 15), torch.randn(2, 2, 17 * 16)]:
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                model(x, torch.tensor([1, 40]))
        for indices in [torch.tensor([1., 40.]), torch.tensor([1]), torch.ones(3, 2, dtype=torch.long), torch.ones(1, 1, 2, dtype=torch.long)]:
            with self.assertRaises(ValueError):model(torch.randn(2, 2, 32), indices)
        for indices in [torch.tensor([0, 40]), torch.tensor([-1, 40]), torch.tensor([129, 40])]:
            with self.assertRaises((IndexError, RuntimeError)):model(torch.randn(2, 2, 32), indices)
        with self.assertRaises(ValueError):model.forward_all(torch.randn(2, 2, 32), torch.tensor([1,40]), normalize=1)

    def test_safe_native_loading_is_complete_explicit_and_round_trips(self):
        model = LaBraMEncoder(**CONFIG).eval()
        state = model.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            for container in [state, {'state_dict': state}, {'model': state}]:
                torch.save(container, path)
                restored = LaBraMEncoder.from_pretrained(path, **CONFIG).eval()
                for name,value in state.items():torch.testing.assert_close(restored.state_dict()[name],value,rtol=0,atol=0)
            native = {'student.' + name:value for name,value in state.items()}
            native['student.mask_token'] = torch.zeros(1,1,16)
            native['student.lm_head.weight'] = torch.zeros(4,16)
            native['other.weight'] = torch.ones(1)
            torch.save({'model':native}, path)
            with self.assertRaises(ValueError):LaBraMEncoder.from_pretrained(path,prefix='student.',**CONFIG)
            restored = LaBraMEncoder.from_pretrained(path,prefix='student.',exclude_pretraining=True,**CONFIG)
            for name,value in state.items():torch.testing.assert_close(restored.state_dict()[name],value,rtol=0,atol=0)
            for bad in [{**state,'norm.weight': torch.ones(15)}, {**state,'norm.weight':torch.full((16,),float('nan'))},
                        {k:v for k,v in state.items() if k!='norm.bias'}, {**state,'extra':torch.ones(1)},
                        {'model':state,'state_dict':state}, {'model':[]}]:
                torch.save(bad,path)
                with self.assertRaises(ValueError):LaBraMEncoder.from_pretrained(path,**CONFIG)

    def test_training_metadata_opt_in_is_scoped_and_default_remains_safe(self):
        from argparse import Namespace
        import numpy as np
        model = LaBraMEncoder(**CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'native.pt'
            torch.save(dict(model=model.state_dict(),args=Namespace(seed=42),score=np.float64(.7)),path)
            with self.assertRaises(Exception):LaBraMEncoder.from_pretrained(path,**CONFIG)
            before = list(torch.serialization.get_safe_globals())
            if not hasattr(torch.serialization,'safe_globals'):
                with self.assertRaises(RuntimeError):LaBraMEncoder.from_pretrained(path,allow_training_metadata=True,**CONFIG)
                return
            LaBraMEncoder.from_pretrained(path,allow_training_metadata=True,**CONFIG)
            self.assertCountEqual(torch.serialization.get_safe_globals(),before)

    @unittest.skipUnless(os.environ.get('GITHUB_ACTIONS') == 'true' or os.environ.get('TORCHEEG_TEST_COMPILE') == '1', 'Optional compiler regression')
    def test_compiled_outputs_gradients_and_channel_bounds(self):
        model = LaBraMEncoder(**CONFIG).eval()
        candidate = copy.deepcopy(model)
        compiled = torch.compile(candidate, backend='inductor', fullgraph=True)
        x = torch.randn(2,2,32,requires_grad=True)
        y = x.detach().clone().requires_grad_()
        indices = torch.tensor([1,40])
        expected = model(x,indices)
        actual = compiled(y,indices)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
        coefficients = torch.linspace(-.7,1.1,actual.numel()).reshape_as(actual)
        (expected*coefficients).sum().backward();(actual*coefficients).sum().backward()
        torch.testing.assert_close(y.grad,x.grad,rtol=2e-3,atol=1e-5)
        for (name,left),(_,right) in zip(candidate.named_parameters(),model.named_parameters()):
            torch.testing.assert_close(left.grad,right.grad,rtol=2e-3,atol=1e-5,msg=name)
        for indices in [torch.tensor([-1,40]),torch.tensor([0,40]),torch.tensor([129,40])]:
            with self.assertRaises((IndexError,RuntimeError)):compiled(y.detach(),indices)
        ablation = torch.compile(LaBraMEncoder(**CONFIG, use_abs_pos_emb=False).eval(), backend='inductor', fullgraph=True)
        for indices in [torch.tensor([-1,40]),torch.tensor([0,40]),torch.tensor([129,40])]:
            with self.assertRaises((IndexError,RuntimeError)):ablation(y.detach(),indices)


if __name__ == '__main__':
    unittest.main()
