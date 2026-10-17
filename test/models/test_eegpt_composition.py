import copy
import tempfile
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from torcheeg.models import (EEGPT, EEGPTReconstructor, EEGPTPredictor, EEGPTFinetuneClassifier,
    EEGPTClassifier, EEGPTRegression, EEGPTEmbedding, EEGPTInputAdapter, EEGPTAdaptedModel,
    add_eegpt_adapters, add_eegpt_prefix, add_eegpt_lora)


def tune(task, method):
    if method in ('adapters', 'combined'):
        add_eegpt_adapters(task, adapter_size=4, dropout=0.)
    if method in ('prefix', 'combined'):
        add_eegpt_prefix(task, prefix_length=2)
    if method in ('lora', 'combined'):
        add_eegpt_lora(task, rank=2, alpha=4., dropout=0.)


class TestEEGPTComposition(unittest.TestCase):
    def task(self, predictor=False, mixing=True, **kwargs):
        encoder = EEGPT(num_electrodes=3, channel_names=['C3','CZ','C4'], chunk_size=128,
                        patch_size=32, hid_channels=16, depth=2, num_heads=4, embed_num=2)
        decoder = (EEGPTPredictor if predictor else EEGPTReconstructor)(num_patches=4,
            in_features=16, hid_channels=16, depth=2, num_heads=4, embed_num=2)
        return EEGPTFinetuneClassifier(encoder,decoder,in_channels=5 if mixing else None,**kwargs)

    def adapter(self, slots=5, samples=128):
        return EEGPTInputAdapter(19,slots,num_samples=samples,temporal_kernel=3,dropout=0.)

    def composed(self, method='none', predictor=False):
        task = self.task(predictor)
        if method != 'none':
            tune(task,method)
        return EEGPTAdaptedModel(self.adapter(),task)

    def test_valid_task_input_slots_differ_from_internal_encoder_slots(self):
        for predictor in (False,True):
            with self.subTest(predictor=predictor):
                task = self.task(predictor)
                model = EEGPTAdaptedModel(self.adapter(),task).eval()
                x = torch.randn(2,19,96)
                self.assertEqual(model(x).shape,(2,2))
                torch.testing.assert_close(model(x),task(model.adapter(x)))
                self.assertEqual(model.feature(x).shape,(2,4,2,16))
                self.assertEqual(model.feature(x,flatten=True).shape,(2,model.feature_dim))

    def test_wrong_task_input_slots_rejected_before_forward(self):
        # Three is the internal encoder width in channels, but the task needs
        # five input features before its explicit pointwise mapping.
        with self.assertRaises(ValueError):
            EEGPTAdaptedModel(self.adapter(slots=3),self.task())
        with self.assertRaises(ValueError):
            EEGPTAdaptedModel(self.adapter(samples=96),self.task(resize_input=True))

    def test_direct_summary_tasks_retain_existing_geometry(self):
        for kind in (EEGPTClassifier,EEGPTRegression,EEGPTEmbedding):
            for slots in (3,5):
                task = self.task(mixing=False)
                direct = kind(task.encoder)
                if slots == 3:
                    model = EEGPTAdaptedModel(self.adapter(slots),direct).eval()
                    self.assertTrue(torch.isfinite(model(torch.randn(2,19,96))).all())
                else:
                    with self.assertRaises(ValueError):
                        EEGPTAdaptedModel(self.adapter(slots),direct)

    def test_all_tuning_methods_train_through_both_full_decoders(self):
        for predictor in (False,True):
            for method in ('adapters','prefix','lora','combined'):
                with self.subTest(predictor=predictor,method=method):
                    model = self.composed(method,predictor).train()
                    frozen = {n:p.detach().clone() for n,p in model.model.encoder.named_parameters() if not p.requires_grad}
                    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
                    x = torch.randn(2,19,96,requires_grad=True)
                    F.cross_entropy(model(x),torch.tensor([0,1])).backward()
                    self.assertGreater(x.grad.norm().item(),0)
                    self.assertGreater(model.adapter.chan_conv[0].weight.grad.norm().item(),0)
                    self.assertGreater(model.model.chan_conv[0].weight.grad.norm().item(),0)
                    self.assertGreater(model.model.decoder.cls_token.grad.norm().item(),0)
                    self.assertGreater(model.model.head.weight.grad.norm().item(),0)
                    tuning = {n:p for n,p in model.model.encoder.named_parameters() if p.requires_grad}
                    self.assertTrue(tuning)
                    for name,p in tuning.items():
                        self.assertIsNotNone(p.grad,name)
                        self.assertTrue(torch.isfinite(p.grad).all(),name)
                    self.assertTrue(any(p.grad.norm().item()>0 for p in tuning.values()))
                    optimizer.step()
                    for name,p in model.model.encoder.named_parameters():
                        if name in frozen:
                            torch.testing.assert_close(p,frozen[name],rtol=0,atol=0)

    def test_frozen_full_backbone_preserves_input_adapter_gradients(self):
        task = self.task(freeze_encoder=True,freeze_decoder=True)
        model = EEGPTAdaptedModel(self.adapter(),task).train()
        self.assertFalse(task.encoder.training)
        self.assertFalse(task.decoder.training)
        self.assertTrue(model.adapter.training)
        x = torch.randn(2,19,96,requires_grad=True)
        F.cross_entropy(model(x),torch.tensor([0,1])).backward()
        self.assertGreater(model.adapter.chan_conv[0].weight.grad.norm().item(),0)
        self.assertTrue(all(p.grad is None for m in (task.encoder,task.decoder) for p in m.parameters()))
        self.assertGreater(x.grad.norm().item(),0)

    def test_complete_composed_tensor_state_roundtrip(self):
        for predictor in (False,True):
            model = self.composed('combined',predictor).eval()
            # Activate initially zero updates so state restoration must retain them.
            with torch.no_grad():
                for name,p in model.model.encoder.named_parameters():
                    if name.endswith('lora_B') or 'up_project.weight' in name:
                        p.normal_(std=.02)
            x = torch.randn(2,19,96)
            expected = model(x)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'composed.pt'
                torch.save(model.state_dict(),path)
                restored = self.composed('combined',predictor).eval()
                restored.load_state_dict(torch.load(path,weights_only=True),strict=True)
            torch.testing.assert_close(restored(x),expected)

    def test_copy_owns_all_tuning_parameters(self):
        original = self.composed('combined')
        clone = copy.deepcopy(original)
        self.assertTrue(all(a.data_ptr()!=b.data_ptr() for a,b in zip(original.parameters(),clone.parameters())))
        clone(torch.randn(2,19,96)).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in original.parameters()))
        self.assertTrue(any(p.grad is not None for p in clone.model.encoder.parameters()))

    def test_evaluation_does_not_update_adapter_statistics(self):
        model = self.composed('prefix').train()
        x = torch.randn(2,19,96)
        model(x)
        statistics = {n:v.clone() for n,v in model.adapter.named_buffers()}
        model.eval()
        torch.testing.assert_close(model(x),model(x),rtol=0,atol=0)
        for name,value in model.adapter.named_buffers():
            torch.testing.assert_close(value,statistics[name],rtol=0,atol=0)
        counters = [m.num_batches_tracked.item() for m in model.adapter.modules() if isinstance(m,nn.BatchNorm2d)]
        self.assertEqual(counters,[1,1])


if __name__ == '__main__':
    unittest.main()
