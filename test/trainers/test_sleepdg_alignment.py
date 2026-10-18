import copy
import hashlib
import io
import json
import os
from pathlib import Path
import unittest

import torch

from torcheeg.trainers import SleepDGAlignmentLoss
from test.models.fixtures.sleepdg_reference.alignment_reference import CORAL


class TestSleepDGAlignmentLoss(unittest.TestCase):
    def test_independent_author_losses_and_all_input_gradients(self):
        cases = [([0, 1, 0, 1, 2, 1, 3, 2], (8, 20, 512)),
                 ([3, 1, 3, 1, 1], (5, 3, 7)), ([2, 0, 1, 2, 0, 1], (6, 4, 5))]
        for codes, shape in cases:
            with self.subTest(codes=codes, shape=shape):
                torch.manual_seed(42)
                x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
                other = x.detach().clone().requires_grad_()
                domains = torch.tensor(codes)
                loss, expected = SleepDGAlignmentLoss()(x, domains), CORAL()(other, domains)
                torch.testing.assert_close(loss, expected, rtol=1e-11, atol=1e-11)
                loss.backward()
                expected.backward()
                torch.testing.assert_close(x.grad, other.grad, rtol=1e-9, atol=1e-11)

    def test_reference_provenance(self):
        root = Path(__file__).parents[1] / 'models/fixtures/sleepdg_reference'
        data = json.loads((root / 'alignment-manifest.json').read_text())
        self.assertEqual(data['revision'], '01d209d4fda874933a277a88bccdca6e3835e884')
        self.assertEqual(data['original_path'], 'losses/double_alignment.py')
        self.assertEqual(hashlib.sha256((root / data['adapted_path']).read_bytes()).hexdigest(),
                         data['adapted_sha256'])

    def test_hand_calculated_pair_reductions_are_not_interchanged(self):
        # Singleton means [0,0], [1,0], [2,0], zero covariances, relations 0,1,1.
        x = torch.tensor([[[0., 0.]], [[1., 0.]], [[2., 0.]]], dtype=torch.float64)
        actual = SleepDGAlignmentLoss(3)(x, torch.arange(3))
        # Means: .5+2+.5=3. Sequence relations: (1+1+0)/3.
        torch.testing.assert_close(actual, torch.tensor(3. + 2./3., dtype=torch.float64))

    def test_covariance_is_unbiased_and_epoch_pairs_are_summed(self):
        x = torch.tensor([[[0.], [2.]], [[0.], [4.]], [[0.], [6.]]], dtype=torch.float64)
        actual = SleepDGAlignmentLoss(3)(x, torch.arange(3))
        # Means 1,2,3; unbiased covariances 2,8,18. H=1 correlations zero.
        expected = (1+36) + (4+256) + (1+100)
        torch.testing.assert_close(actual, torch.tensor(float(expected), dtype=torch.float64))

    def test_missing_domains_general_codes_and_single_domain_zero(self):
        torch.manual_seed(42)
        x = torch.randn(5, 3, 7, dtype=torch.float64, requires_grad=True)
        expected = SleepDGAlignmentLoss(2)(x, torch.tensor([0, 1, 0, 1, 1]))
        actual = SleepDGAlignmentLoss(8)(x, torch.tensor([3, 7, 3, 7, 7]))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for count, code in [(1, 0), (8, 7)]:
            result = SleepDGAlignmentLoss(count)(x, torch.full((5,), code))
            self.assertEqual(result.item(), 0.)
            grad, = torch.autograd.grad(result, x, retain_graph=True)
            torch.testing.assert_close(grad, torch.zeros_like(x), rtol=0, atol=0)

    def test_batch_permutation_domain_relabeling_and_replication(self):
        torch.manual_seed(17)
        x = torch.randn(8, 4, 6, dtype=torch.float64)
        domains = torch.tensor([0, 0, 1, 2, 3, 1, 1, 2])
        loss = SleepDGAlignmentLoss()
        expected = loss(x, domains)
        order = torch.tensor([4, 2, 1, 7, 3, 6, 5, 0])
        torch.testing.assert_close(loss(x[order], domains[order]), expected)
        torch.testing.assert_close(loss(x, 3-domains), expected)
        # Replication changes unbiased covariance because its denominator is n-1.
        self.assertNotEqual(loss(x.repeat_interleave(2, 0), domains.repeat_interleave(2)).item(),
                            expected.item())

    def test_degenerate_features_singletons_and_noncontiguous_inputs(self):
        for shape in [(3, 1, 1), (3, 1, 5), (3, 4, 5)]:
            for fill in (0., 2.):
                with self.subTest(shape=shape, fill=fill):
                    x = torch.full(shape, fill, requires_grad=True)
                    result = SleepDGAlignmentLoss(3)(x, torch.arange(3))
                    result.backward()
                    self.assertEqual(result.item(), 0.)
                    self.assertTrue(torch.isfinite(x.grad).all())
        x = torch.randn(4, 5, 3, dtype=torch.float64).transpose(1, 2).requires_grad_()
        self.assertFalse(x.is_contiguous())
        loss = SleepDGAlignmentLoss()(x, torch.arange(4))
        torch.testing.assert_close(loss, CORAL()(x.contiguous(), torch.arange(4)))
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_numerical_gradient_check_and_input_not_mutated(self):
        torch.manual_seed(13)
        x = torch.randn(4, 2, 3, dtype=torch.float64, requires_grad=True)
        saved = x.detach().clone()
        domains = torch.arange(4)
        self.assertTrue(torch.autograd.gradcheck(SleepDGAlignmentLoss(), (x, domains)))
        torch.testing.assert_close(x.detach(), saved, rtol=0, atol=0)
        torch.testing.assert_close(domains, torch.arange(4), rtol=0, atol=0)

    def test_low_precision_and_autocast_statistics_are_fp32(self):
        torch.manual_seed(42)
        domains = torch.tensor([0, 1, 0, 1])
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(4, 3, 8).to(dtype).requires_grad_()
            reference_x = x.detach().float().requires_grad_()
            expected = SleepDGAlignmentLoss(2)(reference_x, domains)
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                actual = SleepDGAlignmentLoss(2)(x, domains)
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.backward()
            expected.backward()
            torch.testing.assert_close(x.grad, reference_x.grad.to(dtype), rtol=0, atol=0)
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_invalid_configuration_shape_dtype_and_domain_codes(self):
        for value in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                SleepDGAlignmentLoss(num_domains=value)
        for value in (True, 0., -1., float('nan'), float('inf'), 'small'):
            with self.assertRaises(ValueError):
                SleepDGAlignmentLoss(eps=value)
        loss = SleepDGAlignmentLoss(2)
        x, domains = torch.randn(4, 3, 5), torch.tensor([0, 1, 0, 1])
        cases = [(x[0], domains), (x[:, :0], domains), (x[:0], domains[:0]),
                 (x[:, :, :0], domains), (x.long(), domains), (x.to(torch.complex64), domains),
                 (x, domains[:, None]), (x, domains[:2]), (x, domains.float()), (x, domains.bool())]
        for features, codes in cases:
            with self.assertRaises(ValueError):
                loss(features, codes)
        with self.assertRaises(TypeError):
            loss([], domains)
        for codes in (torch.tensor([-1, 1, 0, 1]), torch.tensor([0, 1, 0, 2])):
            with self.assertRaises((RuntimeError, IndexError)):
                loss(x, codes)

    def test_state_round_trip_and_integer_code_dtypes(self):
        loss = SleepDGAlignmentLoss(3, eps=1e-10)
        restored = SleepDGAlignmentLoss(3, eps=1e-10)
        restored.load_state_dict(copy.deepcopy(loss.state_dict()), strict=True)
        x, domains = torch.randn(4, 3, 5), torch.tensor([0, 2, 0, 2])
        for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            torch.testing.assert_close(restored(x, domains.to(dtype)), loss(x, domains))
        self.assertEqual(list(loss.parameters()), [])
        self.assertEqual(list(loss.buffers()), [])

    def test_complete_model_objective_updates_all_training_components(self):
        from torcheeg.models import SleepDG
        from torch.nn import functional as F

        torch.manual_seed(42)
        model = SleepDG(sequence_length=3, hid_channels=8, num_heads=2,
                        mlp_channels=16, dropout=0.0).train()
        initial = {name: p.detach().clone() for name, p in model.named_parameters()}
        x = torch.randn(4, 3, 2, 3000)
        labels = torch.arange(12).reshape(4, 3) % 5
        domains = torch.arange(4)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        alignment = SleepDGAlignmentLoss()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits, reconstruction, features = model.forward_all(x)
            loss = (F.cross_entropy(logits.flatten(0, 1), labels.flatten())
                    + F.mse_loss(reconstruction, x) + alignment(features, domains))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            optimizer.step()
        for name in ('ae.encoder.epoch_encoder.encoder.0.weight',
                     'ae.encoder.seq_encoder.pos_embedding', 'ae.encoder.fc_mu.weight',
                     'ae.decoder.upsample.0.weight', 'ae.decoder.upsample.20.weight',
                     'classifier.weight'):
            self.assertFalse(torch.equal(dict(model.named_parameters())[name], initial[name]), name)

    @unittest.skipUnless(os.getenv('TORCHEEG_TEST_COMPILE') == '1'
                         or os.getenv('GITHUB_ACTIONS') == 'true', 'Optional compiler check')
    def test_fullgraph_compile_gradients_and_export_round_trip(self):
        if not hasattr(torch, 'compile') or not hasattr(torch, 'export'):
            self.skipTest('Requires PyTorch compile and export')
        torch.manual_seed(42)
        loss = SleepDGAlignmentLoss(3)
        x = torch.randn(5, 3, 8, requires_grad=True)
        domains = torch.tensor([0, 1, 2, 0, 2])
        expected = loss(x, domains)
        expected_grad, = torch.autograd.grad(expected, x)
        other = x.detach().clone().requires_grad_()
        compiled = torch.compile(loss, fullgraph=True, dynamic=False)
        actual = compiled(other, domains)
        torch.testing.assert_close(actual, expected)
        actual_grad, = torch.autograd.grad(actual, other)
        torch.testing.assert_close(actual_grad, expected_grad)
        program = torch.export.export(loss, (x.detach(), domains), strict=True)
        buffer = io.BytesIO()
        torch.export.save(program, buffer)
        buffer.seek(0)
        restored = torch.export.load(buffer).module()
        torch.testing.assert_close(restored(x.detach(), domains), expected.detach())
        for invalid in (torch.tensor([0, 1, 3, 0, 2]), torch.tensor([0, -1, 2, 0, 2])):
            with self.assertRaises((RuntimeError, IndexError)):
                compiled(x.detach(), invalid)
            with self.assertRaises((RuntimeError, IndexError)):
                restored(x.detach(), invalid)
        single = SleepDGAlignmentLoss(1)
        one_domain = torch.zeros_like(domains)
        single_compiled = torch.compile(single, fullgraph=True, dynamic=False)
        self.assertEqual(single_compiled(x, one_domain).item(), 0.)
        single_export = torch.export.export(single, (x.detach(), one_domain), strict=True).module()
        self.assertEqual(single_export(x.detach(), one_domain).item(), 0.)
        for invalid in (torch.ones_like(domains), -torch.ones_like(domains)):
            with self.assertRaises((RuntimeError, IndexError)):
                single_compiled(x.detach(), invalid)
            with self.assertRaises((RuntimeError, IndexError)):
                single_export(x.detach(), invalid)



if __name__ == '__main__':
    unittest.main()
