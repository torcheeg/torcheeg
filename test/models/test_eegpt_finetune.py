import copy
import gc
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from torcheeg.models import EEGPT, EEGPTReconstructor, EEGPTPredictor, EEGPTFinetuneClassifier
from test.models.test_eegpt import reference_state
from test.models import test_eegpt_decoder as decoder_reference


class UnsupportedCheckpointObject:
    pass


class TestEEGPTFinetune(unittest.TestCase):
    def encoder_config(self, **kwargs):
        config = dict(num_electrodes=3, channel_names=['C3', 'CZ', 'C4'], chunk_size=128,
                      patch_size=32, hid_channels=16, depth=2, num_heads=4, embed_num=2)
        config.update(kwargs)
        return config

    def decoder_config(self, **kwargs):
        config = dict(num_patches=4, in_features=16, embed_num=2, hid_channels=16,
                      depth=2, num_heads=4, patch_size=8)
        config.update(kwargs)
        return config

    def components(self, predictor=False, **kwargs):
        encoder = EEGPT(**self.encoder_config())
        config = self.decoder_config(**kwargs)
        if predictor:
            config.pop('patch_size')
        decoder = (EEGPTPredictor if predictor else EEGPTReconstructor)(**config)
        return encoder, decoder

    def model(self, predictor=False, **kwargs):
        return EEGPTFinetuneClassifier(*self.components(predictor), **kwargs)

    def test_canonical_official_full_model_outputs_and_gradients(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures' / 'eegpt_finetune_reference.json').read_text())
        self.assertEqual(fixture['commit'], 'ea7282f698c406cd423ccbedbe9960853761144c')
        self.assertEqual(fixture['source_sha256'], '0371411f4c2bfc0d1d0999432c8cea82e1d0f76d2e89ac95089adfeb59ed2233')
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(fixture['threads'])
            for case in fixture['cases']:
                config = case['parameters']
                predictor = config['use_predictor']
                with self.subTest(predictor=predictor):
                    encoder = EEGPT(num_electrodes=3, channel_names=config['use_channels_names'], chunk_size=128)
                    decoder = (EEGPTPredictor if predictor else EEGPTReconstructor)(num_patches=2)
                    model = EEGPTFinetuneClassifier(encoder, decoder, num_classes=3,
                        mean_pooling=config['use_mean_pooling'], use_out_proj=config['use_out_proj'],
                        in_channels=config['in_channels'] if config['use_chan_conv'] else None,
                        resize_input=True).double().eval()
                    def native_name(name):
                        if name.startswith('encoder.'):
                            return 'target_encoder.' + name[len('encoder.'):]
                        if name.startswith('decoder.'):
                            return ('predictor.' if predictor else 'reconstructor.') + name[len('decoder.'):]
                        return name
                    with torch.no_grad():
                        for name, value in model.state_dict().items():
                            if not name.endswith('time_embed.freqs'):
                                value.copy_(reference_state(native_name(name), value))
                    shape = case['input_shape']
                    values = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float64)
                    eeg = (torch.sin(values / 17) + torch.cos(values / 31)).reshape(shape).requires_grad_()
                    output = model(eeg)
                    torch.testing.assert_close(output, torch.tensor(case['output'], dtype=torch.float64), rtol=1e-9, atol=1e-9)
                    coefficients = torch.sin(torch.arange(output.numel(), dtype=torch.float64) / 13).reshape(output.shape)
                    (output * coefficients).square().mean().backward()
                    decoder_reference.TestEEGPTDecoder.assert_summary(self, eeg.grad, case['input_gradient'])
                    self.assertEqual({native_name(n) for n,_ in model.named_parameters()}, set(case['parameter_gradients']))
                    for name, parameter in model.named_parameters():
                        with self.subTest(parameter=name):
                            decoder_reference.TestEEGPTDecoder.assert_summary(self, parameter.grad, case['parameter_gradients'][native_name(name)])
                del model, encoder, decoder, eeg, output, values, parameter, value
                gc.collect()
        finally:
            torch.set_num_threads(previous)

    def test_pooling_order_and_token_modes(self):
        x = torch.randn(2, 3, 128)
        for predictor in (False, True):
            for mean in (False, True):
                for projected in (False, True):
                    with self.subTest(predictor=predictor, mean=mean, projected=projected):
                        model = self.model(predictor, mean_pooling=mean, use_out_proj=projected).eval()
                        tokens = model.decoder(model.encoder(x), use_out_proj=projected)
                        if mean:
                            expected = model.fc_norm(tokens[:, 1:].mean(1))
                            all_tokens = model.fc_norm(tokens)
                        else:
                            all_tokens = model.norm(tokens)
                            expected = all_tokens[:, 0]
                        torch.testing.assert_close(model.forward_features(x), expected)
                        torch.testing.assert_close(model.forward_features(x, return_all_tokens=True), all_tokens)
                        torch.testing.assert_close(model.forward_features(x, return_patch_tokens=True), all_tokens[:, 1:])
                        self.assertEqual(model(x).shape, (2, 2))
                        self.assertEqual(expected.shape, (2, model.decoder_feature_dim))
                        torch.testing.assert_close(model.feature(x), model.encoder(x))
                        self.assertEqual(model.feature(x, flatten=True).shape, (2, model.feature_dim))

    def test_zero_class_embedding_head(self):
        for predictor in (False, True):
            model = self.model(predictor, num_classes=0).eval()
            x = torch.randn(2, 3, 128)
            self.assertIsInstance(model.head, nn.Identity)
            torch.testing.assert_close(model(x), model.forward_features(x))

    def test_optional_mixing_and_explicit_nearest_resize(self):
        model = self.model(in_channels=5, resize_input=True).eval()
        x = torch.randn(2, 5, 5, 32)
        torch.testing.assert_close(model.feature(x), model.encoder(model.chan_conv(F.interpolate(x.flatten(2), 128, mode='nearest'))))
        self.assertEqual(model(x).shape, (2, 2))
        strict = self.model()
        with self.assertRaises(ValueError):
            strict(torch.randn(2, 3, 160))

    def test_input_gradients_through_frozen_components(self):
        model = self.model(freeze_encoder=True, freeze_decoder=True, in_channels=5)
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.decoder.training)
        self.assertTrue(model.head.training)
        x = torch.randn(2, 5, 128, requires_grad=True)
        model(x).square().mean().backward()
        self.assertGreater(x.grad.norm().item(), 0)
        self.assertGreater(model.chan_conv[0].weight.grad.norm().item(), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.decoder.parameters()))
        self.assertIsNotNone(model.head.weight.grad)

    def test_frozen_decoder_class_token_training(self):
        model = self.model(freeze_encoder=True, freeze_decoder=True, train_cls_token=True)
        model.train()
        x = torch.randn(2, 3, 128)
        model(x).square().mean().backward()
        active = [name for name,p in model.decoder.named_parameters() if p.requires_grad]
        self.assertEqual(active, ['cls_token'])
        self.assertGreater(model.decoder.cls_token.grad.norm().item(), 0)
        self.assertFalse(model.decoder.training)

    def test_existing_flags_remain_when_freezing_disabled(self):
        encoder, decoder = self.components()
        encoder.summary_token.requires_grad_(False)
        decoder.cls_token.requires_grad_(False)
        before = {id(p):p.requires_grad for m in (encoder,decoder) for p in m.parameters()}
        model = EEGPTFinetuneClassifier(encoder, decoder)
        self.assertEqual(before, {id(p):p.requires_grad for m in (model.encoder,model.decoder) for p in m.parameters()})

    def test_mixing_constraints_keep_optimizer_and_backward_ownership(self):
        model = self.model(in_channels=5, max_norm=.2, channel_max_norm=.3)
        parameter = model.chan_conv[0].weight
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        pointer = parameter.data_ptr()
        with torch.no_grad():
            parameter.mul_(100)
            model.head.weight.mul_(100)
        x = torch.randn(2, 5, 128, requires_grad=True)
        loss = model(x).square().mean() + model(x).square().mean()
        loss.backward()
        self.assertIs(parameter, model.chan_conv[0].weight)
        self.assertEqual(pointer, parameter.data_ptr())
        self.assertTrue(any(p is parameter for group in optimizer.param_groups for p in group['params']))
        self.assertLessEqual(parameter.flatten(1).norm(dim=1).max().item(), .300001)
        self.assertLessEqual(model.head.weight.norm(dim=1).max().item(), .200001)
        self.assertGreater(parameter.grad.norm().item(), 0)
        optimizer.step()

    def test_frozen_constraint_forward_preserves_state(self):
        model = self.model(in_channels=5, channel_max_norm=.3).requires_grad_(False).eval()
        with torch.no_grad():
            model.chan_conv[0].weight.mul_(100)
        before = copy.deepcopy(model.state_dict())
        x = torch.randn(2, 5, 128, requires_grad=True)
        model(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_dtype_copy_and_complete_task_tensor_roundtrip(self):
        for predictor in (False, True):
            model = self.model(predictor, in_channels=5, resize_input=True).double().eval()
            x = torch.randn(2, 5, 160, dtype=torch.float64)
            expected = model(x)
            clone = copy.deepcopy(model)
            self.assertNotEqual(clone.encoder.summary_token.data_ptr(), model.encoder.summary_token.data_ptr())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'task.pt'
                torch.save(model.state_dict(), path)
                clone.load_state_dict(torch.load(path, weights_only=True), strict=True)
            torch.testing.assert_close(clone(x), expected)
            clone(x).sum().backward()
            self.assertIsNone(model.encoder.summary_token.grad)
            self.assertIsNotNone(clone.encoder.summary_token.grad)

    def checkpoint(self, predictor=False):
        encoder, decoder = self.components(predictor)
        prefix = 'predictor.' if predictor else 'reconstructor.'
        state = {'target_encoder.'+n:v.clone() for n,v in encoder.state_dict().items()}
        state.update({prefix+n:v.clone() for n,v in decoder.state_dict().items()})
        state['unrelated_projection.weight'] = torch.zeros(2)
        return encoder, decoder, {'state_dict':state}

    def load(self, path, predictor=False, **kwargs):
        config = self.decoder_config()
        if predictor:
            config.pop('patch_size')
        return EEGPTFinetuneClassifier.from_pretrained(path, self.encoder_config(), config,
            decoder_type='predictor' if predictor else 'reconstructor', **kwargs)

    def test_complete_joint_component_loader_uses_one_safe_read(self):
        for predictor in (False, True):
            encoder, decoder, state = self.checkpoint(predictor)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'joint.pt'
                torch.save(state, path)
                with patch('torch.load', wraps=torch.load) as loader:
                    model = self.load(path, predictor)
                loader.assert_called_once()
                self.assertTrue(loader.call_args.kwargs['weights_only'])
                self.assertEqual(loader.call_args.kwargs['map_location'], 'cpu')
            for expected, actual in [(encoder.state_dict(),model.encoder.state_dict()), (decoder.state_dict(),model.decoder.state_dict())]:
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key])
            self.assertTrue(torch.isfinite(model(torch.randn(2,3,128))).all())

    def test_explicit_component_prefixes_and_plain_outer_state(self):
        encoder, decoder, state = self.checkpoint()
        remapped = {n.replace('target_encoder.', 'model.encoder.').replace('reconstructor.', 'model.decoder.'):v
                    for n,v in state['state_dict'].items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'joint.pt'
            torch.save(remapped, path)
            model = self.load(path, encoder_prefix='model.encoder.', decoder_prefix='model.decoder.')
        torch.testing.assert_close(model.encoder.summary_token, encoder.summary_token)
        torch.testing.assert_close(model.decoder.cls_token, decoder.cls_token)

    def test_loader_rejects_incomplete_extra_or_invalid_component_tensors(self):
        _, _, original = self.checkpoint()
        for prefix, suffix in [('target_encoder.', 'summary_token'), ('reconstructor.', 'cls_token')]:
            key = prefix + suffix
            for defect in ['missing','extra','shape','int','complex','non_tensor','nan','inf']:
                state = copy.deepcopy(original)
                values = state['state_dict']
                if defect == 'missing': del values[key]
                elif defect == 'extra': values[prefix+'unexpected'] = torch.zeros(1)
                elif defect == 'shape': values[key] = torch.zeros(1)
                elif defect == 'int': values[key] = values[key].long()
                elif defect == 'complex': values[key] = values[key].to(torch.complex64)
                elif defect == 'non_tensor': values[key] = 'invalid'
                else: values[key].fill_(float(defect))
                with self.subTest(prefix=prefix, defect=defect), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'joint.pt'
                    torch.save(state, path)
                    with self.assertRaises(ValueError):
                        self.load(path)

    def test_loader_rejects_unsupported_pickle_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'unsafe.pt'
            torch.save({'state_dict':{}, 'object':UnsupportedCheckpointObject()}, path)
            with patch('torch.load', wraps=torch.load) as loader, self.assertRaises(pickle.UnpicklingError):
                self.load(path)
            self.assertEqual(loader.call_count, 1)
            self.assertTrue(loader.call_args.kwargs['weights_only'])

    def test_loader_rejects_invalid_outer_state_and_prefixes(self):
        for value in ([], {'state_dict':[]}, {'state_dict':{1:torch.ones(1)}}):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'bad.pt'
                torch.save(value, path)
                with self.assertRaises(ValueError):
                    self.load(path)
        for kwargs in [dict(encoder_prefix=''), dict(decoder_prefix=''), dict(encoder_prefix=None),
                       dict(decoder_prefix=1), dict(encoder_prefix='a',decoder_prefix='ab')]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.load('not-read.pt', **kwargs)
        for encoder, decoder, kind in [([],{},'reconstructor'), ({},[],'reconstructor'), ({},{},'unknown')]:
            with self.assertRaises(ValueError):
                EEGPTFinetuneClassifier.from_pretrained('not-read.pt', encoder, decoder, decoder_type=kind)

    def test_invalid_combinations_and_constructor_parameters(self):
        for kwargs in [dict(num_classes=-1), dict(num_classes=True), dict(num_classes=1.), dict(max_norm=0),
                       dict(channel_max_norm=float('inf')),dict(norm_eps=0),dict(norm_eps=float('nan')),
                       dict(mean_pooling=1),dict(use_out_proj=1),dict(freeze_decoder=1),dict(freeze_encoder=1),
                       dict(train_cls_token=True),dict(resize_input='yes'),dict(in_channels=0)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model(**kwargs)
        encoder, decoder = self.components()
        with self.assertRaises(ValueError): EEGPTFinetuneClassifier(nn.Identity(),decoder)
        with self.assertRaises(ValueError): EEGPTFinetuneClassifier(encoder,nn.Identity())
        for kwargs in [dict(num_patches=3),dict(embed_num=3),dict(in_features=12),dict(return_attention_layer=1)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EEGPTFinetuneClassifier(*self.components(**kwargs))
        with self.assertRaises(ValueError): EEGPTFinetuneClassifier(encoder,decoder.double())

    def test_invalid_inputs_and_token_settings(self):
        model = self.model()
        for value in (None, [], torch.randn(2,128),torch.randn(2,4,128),torch.randn(2,3,1,1,128),torch.randn(2,3,0)):
            with self.subTest(value_type=type(value)), self.assertRaises(ValueError):
                model(value)
        x = torch.randn(2,3,128)
        for kwargs in [dict(return_patch_tokens=1),dict(return_all_tokens=1),dict(return_all_tokens=True,return_patch_tokens=True)]:
            with self.assertRaises(ValueError): model.forward_features(x,**kwargs)
        with self.assertRaises(ValueError): model.feature(x,flatten=1)


if __name__ == '__main__':
    unittest.main()
