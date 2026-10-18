import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.signal import resample
import torch

from torcheeg.models import LaBraMEncoder, LaBraMClassifier, LaBraMRegression, LaBraMEmbedding


CONFIG = dict(patch_size=16, hid_channels=16, conv_channels=8, mlp_channels=32, depth=2, num_heads=4)
NAMES = ['FP1', 'C3']


def tensor(record):
    return torch.tensor(record['values'], dtype=torch.float64).reshape(record['shape'])


class TestLaBraMHeads(unittest.TestCase):
    def test_native_encoder_independent_head_and_scipy_resampling_gradients(self):
        reference = json.loads((Path(__file__).parent / 'fixtures' / 'labram_head_reference.json').read_text())
        for case in reference['cases']:
            with self.subTest(mode=case['mode']):
                backbone = LaBraMEncoder(**CONFIG).double().eval()
                backbone.load_state_dict({n:tensor(v) for n,v in reference['encoder_state'].items()},strict=True)
                model = LaBraMClassifier(backbone,NAMES,48,num_classes=3,sampling_rate=150,resample_mode=case['mode'],dropout=0.).eval()
                model.task_head.load_state_dict({n:tensor(v) for n,v in reference['head_state'].items()},strict=True)
                x = tensor(case['input']).requires_grad_()
                torch.testing.assert_close(model.preprocess(x),tensor(case['preprocessed']),rtol=1e-11,atol=1e-12)
                torch.testing.assert_close(model.feature(x),tensor(case['features']),rtol=1e-10,atol=1e-11)
                actual = model(x)
                torch.testing.assert_close(actual,tensor(case['output']),rtol=1e-10,atol=1e-11)
                (actual*tensor(case['coefficient'])).sum().backward()
                torch.testing.assert_close(x.grad,tensor(case['input_gradient']),rtol=1e-9,atol=1e-10)
                self.assertEqual(set(dict(backbone.named_parameters())),set(case['encoder_gradients']))
                for name,p in backbone.named_parameters():
                    expected = case['encoder_gradients'][name]
                    if expected is None:self.assertIsNone(p.grad,name)
                    else:torch.testing.assert_close(p.grad,tensor(expected),rtol=1e-9,atol=1e-10,msg=name)
                for name,p in model.task_head.named_parameters():
                    torch.testing.assert_close(p.grad,tensor(case['head_gradients'][name]),rtol=1e-9,atol=1e-10,msg=name)

    def test_window_and_patch_modes_have_independent_periodic_boundary_contracts(self):
        x = torch.arange(2*2*48,dtype=torch.float64).reshape(2,2,48)
        whole = LaBraMEmbedding(LaBraMEncoder(**CONFIG).double(),NAMES,48,sampling_rate=150)
        segments = LaBraMEmbedding(LaBraMEncoder(**CONFIG).double(),NAMES,48,sampling_rate=150,resample_mode='patch')
        np.testing.assert_allclose(whole.preprocess(x).numpy(),resample(x.numpy(),64,axis=-1),rtol=1e-12,atol=1e-12)
        expected = resample(x.numpy().reshape(2,2,4,12),16,axis=-1).reshape(2,2,64)
        np.testing.assert_allclose(segments.preprocess(x).numpy(),expected,rtol=1e-12,atol=1e-12)
        self.assertGreater((whole.preprocess(x)-segments.preprocess(x)).abs().max().item(),1.)
        native = LaBraMEmbedding(LaBraMEncoder(**CONFIG),NAMES,64)
        signal = torch.randn(2,2,64)
        self.assertIs(native.preprocess(signal),signal)

    def test_grid_flatten_and_raw_normalized_feature_modes(self):
        model = LaBraMEmbedding(LaBraMEncoder(**CONFIG),NAMES,64,flatten=False).eval()
        x = torch.randn(2,2,64)
        grid = model(x)
        self.assertEqual(grid.shape,(2,2,4,16))
        torch.testing.assert_close(model.feature(x,flatten=True),grid.flatten(1))
        model.flatten = True
        torch.testing.assert_close(model(x),grid.flatten(1))
        model.normalize_features = False
        raw = model(x)
        torch.testing.assert_close(raw,model.backbone.forward_features(x,model.channel_indices,normalize=False).flatten(1))
        self.assertGreater((raw-grid.flatten(1)).abs().max().item(),1e-5)
        with self.assertRaises(ValueError):model.feature(x,flatten=1)

    def test_head_topologies_dropout_pooling_and_scalar_output_axis(self):
        for patches in [2,6]:
            model = LaBraMClassifier(LaBraMEncoder(**CONFIG),NAMES,patches*16,dropout=.37)
            self.assertEqual(model.head_channels,(max(4,patches)*16,16))
            self.assertEqual(model.task_head.linear0.in_features,2*patches*16)
            self.assertEqual(model.task_head.dropout0.p,.37)
        model = LaBraMRegression(LaBraMEncoder(**CONFIG),NAMES,64,pooling='mean',head_channels=(),num_outputs=3)
        self.assertEqual(len(model.task_head),1)
        with torch.no_grad():model.task_head.output.weight.zero_();model.task_head.output.bias.fill_(-2.)
        torch.testing.assert_close(model(torch.randn(2,2,64)),torch.full((2,3),-2.))
        scalar = LaBraMRegression(LaBraMEncoder(**CONFIG),NAMES,64)
        self.assertEqual(scalar(torch.randn(1,2,64)).shape,(1,1))
        custom = LaBraMClassifier(LaBraMEncoder(**CONFIG),NAMES,64,head_channels=(7,))
        self.assertEqual(custom.task_head.output.in_features,7)

    def test_freeze_keeps_eval_input_gradients_and_two_real_head_updates(self):
        model = LaBraMClassifier(LaBraMEncoder(**CONFIG,dropout=.3),NAMES,64,freeze_encoder=True)
        state = {n:v.clone() for n,v in model.backbone.state_dict().items()}
        before = model.task_head.output.weight.detach().clone()
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.task_head.training)
        x = torch.randn(2,2,64,requires_grad=True)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.1)
        for _ in range(2):
            optimizer.zero_grad();x.grad=None
            torch.nn.functional.cross_entropy(model(x),torch.tensor([0,1])).backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().sum().item(),0.)
            optimizer.step()
        self.assertFalse(torch.equal(before,model.task_head.output.weight))
        for n,v in model.backbone.state_dict().items():torch.testing.assert_close(v,state[n],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

    def test_parameter_flags_and_validation_are_atomic(self):
        backbone = LaBraMEncoder(**CONFIG)
        next(backbone.parameters()).requires_grad_(False)
        flags = {n:p.requires_grad for n,p in backbone.named_parameters()}
        LaBraMClassifier(backbone,NAMES,64,freeze_encoder=False)
        self.assertEqual(flags,{n:p.requires_grad for n,p in backbone.named_parameters()})
        for kwargs in [dict(num_classes=0),dict(sampling_rate=0),dict(sampling_rate=128,chunk_size=33),
                       dict(chunk_size=63),dict(chunk_size=17*16),dict(resample_mode='typo'),
                       dict(sampling_rate=160,chunk_size=64,resample_mode='patch'),dict(dropout=1.),
                       dict(head_channels=(0,)),dict(head_channels='linear'),dict(pooling='typo'),
                       dict(normalize_features=1),dict(freeze_encoder=1),dict(channel_names=['unknown'])]:
            settings = dict(channel_names=NAMES,chunk_size=64,freeze_encoder=True)
            settings.update(kwargs)
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):LaBraMClassifier(backbone,**settings)
            self.assertEqual(flags,{n:p.requires_grad for n,p in backbone.named_parameters()})
        with self.assertRaises(ValueError):LaBraMEmbedding(backbone,NAMES,64,flatten=1)
        interpolated = LaBraMEmbedding(LaBraMEncoder(**CONFIG,max_patches=2,interpolate_time=True),NAMES,64)
        self.assertEqual(interpolated.num_patches,4)
        self.assertTrue(torch.isfinite(interpolated(torch.randn(2,2,64))).all())

    def test_source_shape_dtype_and_task_state_round_trip(self):
        model = LaBraMClassifier(LaBraMEncoder(**CONFIG).double(),NAMES,64).eval()
        for x in [torch.randn(2,3,64),torch.randn(2,2,63),torch.randn(0,2,64),torch.randn(2,2,4,16),
                  torch.ones(2,2,64,dtype=torch.long),torch.ones(2,2,64,dtype=torch.complex64)]:
            with self.assertRaises(ValueError):model(x)
        self.assertEqual(model.channel_indices.dtype,torch.long)
        self.assertEqual(model.task_head.output.weight.dtype,torch.float64)
        restored = LaBraMClassifier(LaBraMEncoder(**CONFIG).double(),NAMES,64).eval()
        restored.load_state_dict(model.state_dict(),strict=True)
        x = torch.randn(2,2,64,dtype=torch.float64)
        torch.testing.assert_close(restored(x),model(x),rtol=0,atol=0)
        torch.testing.assert_close(copy.deepcopy(model)(x),model(x),rtol=0,atol=0)

    def test_safe_complete_encoder_factory_initializes_fresh_task_head(self):
        backbone = LaBraMEncoder(**CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'encoder.pt';torch.save(backbone.state_dict(),path)
            model = LaBraMClassifier.from_pretrained(path,backbone_kwargs=CONFIG,channel_names=NAMES,chunk_size=64)
            for n,v in backbone.state_dict().items():torch.testing.assert_close(model.backbone.state_dict()[n],v,rtol=0,atol=0)
            with self.assertRaises(ValueError):LaBraMClassifier.from_pretrained(path,backbone_kwargs=[],channel_names=NAMES,chunk_size=64)

    @unittest.skipUnless(os.environ.get('GITHUB_ACTIONS')=='true' or os.environ.get('TORCHEEG_TEST_COMPILE')=='1','Optional compiler regression')
    def test_compiled_window_and_patch_modes_match_all_gradients(self):
        for mode in ['window','patch']:
            reference = LaBraMClassifier(LaBraMEncoder(**CONFIG),NAMES,48,sampling_rate=150,resample_mode=mode,dropout=0.).eval()
            candidate = copy.deepcopy(reference)
            x = torch.randn(2,2,48,requires_grad=True);y=x.detach().clone().requires_grad_()
            expected = reference(x)
            actual = torch.compile(candidate,backend='inductor',fullgraph=True,dynamic=False)(y)
            coefficients = torch.tensor([[.2,-.3],[-.4,.7]])
            (expected*coefficients).sum().backward();(actual*coefficients).sum().backward()
            torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
            torch.testing.assert_close(y.grad,x.grad,rtol=2e-3,atol=1e-5)
            for (name,left),(_,right) in zip(candidate.named_parameters(),reference.named_parameters()):
                torch.testing.assert_close(left.grad,right.grad,rtol=2e-3,atol=1e-5,msg=name)


if __name__=='__main__':unittest.main()
