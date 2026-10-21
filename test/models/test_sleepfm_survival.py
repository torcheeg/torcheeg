"""Independent survival math/protocol tests; full author models are separate."""

import copy
import io
import unittest

import torch
from torch import nn

from torcheeg.models import SleepFMDiagnosis, SleepFMSurvivalObjective
from torcheeg.models.transformer.sleepfm_survival import _breslow_loss


class DiagnosisProtocol(nn.Module):
    """Independent input/output fixture, not a replacement pretrained model."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(130, 1065)

    def forward(self, x, mask, demographics):
        weights = (~mask).to(x.dtype).unsqueeze(-1)
        features = (x * weights).sum((1, 2)) / weights.sum((1, 2))
        return self.head(torch.cat((features, demographics), dim=-1))


def risk_set_reference(log_risks, times, events):
    """Enumerate all subjects at risk instead of sorting/prefix scanning."""
    terms = []
    for column in range(log_risks.shape[1]):
        selected = torch.where(events[:, column].bool())[0]
        if selected.numel() == 0:
            terms.append((log_risks[:, column] * 0).sum())
        else:
            values = []
            for row in selected:
                at_risk = times[:, column] >= times[row, column]
                values.append(
                    log_risks[at_risk, column].logsumexp(0) - log_risks[row, column]
                )
            terms.append(torch.stack(values).mean())
    return torch.stack(terms).mean()


def objective(ties='breslow', native_criterion=None, criterion_id=None, names=None):
    return SleepFMSurvivalObjective(
        SleepFMDiagnosis(
            DiagnosisProtocol(), ['BAS', 'RESP'], ['encoded_age', 'encoded_gender']
        ),
        names or ['endpoint_' + str(i) for i in range(1065)],
        ties=ties,
        native_criterion=native_criterion,
        criterion_id=criterion_id,
    )


class TestSleepFMSurvival(unittest.TestCase):
    def test_fractional_followup_precision_and_cpu_autocast_accumulation(self):
        risks = torch.tensor([[2.0], [3.0], [5.0]], dtype=torch.float64).log()
        times = torch.tensor([[2.0], [2.0 + 1e-10], [1.0]], dtype=torch.float64)
        events = torch.tensor([[1], [1], [0]])
        actual = _breslow_loss(risks, times, events)
        torch.testing.assert_close(
            actual, risk_set_reference(risks, times, events), rtol=0, atol=1e-15
        )
        self.assertNotEqual(
            actual.item(), _breslow_loss(risks, times.float(), events).item()
        )
        model = objective()
        x, demo = torch.randn(3, 2, 3, 128), torch.randn(3, 2)
        full_times = times.expand(3, 1065)
        full_events = events.expand(3, 1065)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            out = model(x, demo, full_times, full_events)
        self.assertEqual(out['log_risks'].dtype, torch.bfloat16)
        self.assertEqual(out['loss'].dtype, torch.float32)
        expected = risk_set_reference(out['log_risks'].float(), full_times, full_events)
        torch.testing.assert_close(out['loss'], expected, rtol=1e-6, atol=1e-7)
        out['loss'].backward()
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
        )

    def test_independent_breslow_risk_sets_losses_and_gradients(self):
        torch.manual_seed(19)
        left = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        right = left.detach().clone().requires_grad_()
        times = torch.tensor(
            [[2, 1, 0, 5], [2, 3, 0, 5], [1, 2, 0, 4], [3, 3, 0, 3], [2, 4, 0, 2]],
            dtype=torch.float64,
        )
        events = torch.tensor(
            [[1, 1, 0, 1], [1, 0, 0, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1]]
        )
        actual = _breslow_loss(left, times, events)
        expected = risk_set_reference(right, times, events)
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-13, atol=1e-13)
        self.assertTrue(
            torch.autograd.gradcheck(lambda x: _breslow_loss(x, times, events), (left,))
        )
        self.assertEqual(torch.count_nonzero(left.grad[:, 2]), 0)

    def test_tied_events_censoring_permutation_and_column_shifts(self):
        # Both events at time2 share risk weights2+3+5. The same-time
        # censored subject stays at risk; the earlier censored subject does not.
        risks = (
            torch.tensor([2.0, 3.0, 5.0, 7.0], dtype=torch.float64).log().unsqueeze(1)
        )
        times = torch.tensor([2.0, 2.0, 2.0, 1.0]).unsqueeze(1)
        events = torch.tensor([True, True, False, False]).unsqueeze(1)
        expected = (
            -(
                torch.log(torch.tensor(0.2, dtype=torch.float64))
                + torch.log(torch.tensor(0.3, dtype=torch.float64))
            )
            / 2
        )
        loss = _breslow_loss(risks, times, events)
        torch.testing.assert_close(loss, expected, rtol=0, atol=1e-15)
        for order in [torch.tensor([2, 1, 0, 3]), torch.tensor([3, 0, 2, 1])]:
            torch.testing.assert_close(
                _breslow_loss(risks[order], times[order], events[order]),
                loss,
                rtol=0,
                atol=1e-15,
            )
        torch.testing.assert_close(
            _breslow_loss(risks + 1000, times, events), loss, rtol=0, atol=1e-12
        )
        extreme = torch.tensor([[10000.0], [-10000.0], [9999.0]], requires_grad=True)
        value = _breslow_loss(
            extreme, torch.ones_like(extreme), torch.ones_like(extreme)
        )
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_complete_columns_inputs_demographics_and_optimizer(self):
        torch.manual_seed(7)
        model = objective().double()
        x = torch.randn(4, 2, 3, 128, dtype=torch.float64, requires_grad=True)
        demo = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        times = (
            torch.arange(4, dtype=torch.float64).unsqueeze(1).expand(4, 1065).clone()
        )
        events = torch.ones(4, 1065, dtype=torch.bool)
        mask = torch.zeros(4, 2, 3, dtype=torch.bool)
        mask[1, 1] = True
        mask[1, :, -1] = True
        before = model.model.backend.head.weight.detach().clone()
        out = model(x, demo, times, events, mask)
        self.assertEqual(out['log_risks'].shape, (4, 1065))
        expected = risk_set_reference(out['log_risks'], times, events)
        torch.testing.assert_close(out['loss'], expected, rtol=1e-13, atol=1e-13)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)
        out['loss'].backward()
        self.assertTrue((model.model.backend.head.weight.grad.abs().sum(1) > 0).all())
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(demo.grad).all())
        self.assertEqual(torch.count_nonzero(x.grad[1, 1]), 0)
        self.assertEqual(torch.count_nonzero(x.grad[1, :, -1]), 0)
        opt.step()
        self.assertFalse(torch.equal(before, model.model.backend.head.weight))

    def test_single_subject_all_censored_and_zero_event_endpoint_mean(self):
        extreme = torch.tensor([[3e38], [-3e38]], requires_grad=True)
        zero = _breslow_loss(extreme, torch.ones(2, 1), torch.zeros(2, 1))
        self.assertEqual(zero.item(), 0)
        zero.backward()
        self.assertEqual(torch.count_nonzero(extreme.grad), 0)
        for batch in [1, 3]:
            risks = torch.randn(batch, 2, dtype=torch.float64, requires_grad=True)
            times = torch.arange(batch).unsqueeze(1).expand(batch, 2)
            events = (
                torch.zeros(batch, 2, dtype=torch.bool)
                if batch > 1
                else torch.ones(batch, 2, dtype=torch.bool)
            )
            loss = _breslow_loss(risks, times, events)
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertEqual(torch.count_nonzero(risks.grad), 0)
        risks = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.float64)
        times = torch.tensor([[1, 1], [2, 2]])
        events = torch.tensor([[1, 0], [0, 0]])
        expected = risks[:, 0].logsumexp(0) / 2
        torch.testing.assert_close(_breslow_loss(risks, times, events), expected)

    def test_native_delegation_identity_and_malformed_return(self):
        calls = []

        def explicit_criterion(log_risks, times, events):
            calls.append((log_risks, times, events))
            return log_risks.square().mean()

        model = objective('native', explicit_criterion, 'independent_protocol_only')
        x = torch.randn(2, 2, 3, 128)
        demo = torch.randn(2, 2)
        times, events = torch.ones(2, 1065), torch.ones(2, 1065, dtype=torch.bool)
        out = model(x, demo, times, events)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], out['log_risks'])
        self.assertIs(calls[0][1], times)
        self.assertIs(calls[0][2], events)
        torch.testing.assert_close(out['loss'], out['log_risks'].square().mean())
        for value in [
            torch.tensor(float('nan')),
            torch.ones(2),
            1.0,
            torch.ones((), dtype=torch.long),
        ]:
            bad = objective('native', lambda *args: value, 'invalid_return_protocol')
            with self.assertRaises(ValueError):
                bad(x, demo, times, events)
        changed_source = objective('native', explicit_criterion, 'changed_source')
        with self.assertRaisesRegex(RuntimeError, 'criterion identity'):
            changed_source.load_state_dict(model.state_dict(), strict=True)

    def test_target_validation_precedes_model_and_constructor_guards(self):
        model = objective()
        x, demo = torch.randn(2, 2, 3, 128), torch.zeros(2, 2)
        times, events = torch.ones(2, 1065), torch.ones(2, 1065)
        calls = []
        hook = model.model.register_forward_pre_hook(lambda *args: calls.append(1))
        self.addCleanup(hook.remove)
        for bad in [
            None,
            times[:, :-1],
            times.bool(),
            times * -1,
            times * float('nan'),
            times.to(torch.complex64),
        ]:
            with self.assertRaises(ValueError):
                model(x, demo, bad, events)
        for bad in [
            None,
            events[:, :-1],
            events * 2,
            events * float('inf'),
            events.to(torch.complex64),
        ]:
            with self.assertRaises(ValueError):
                model(x, demo, times, bad)
        self.assertEqual(calls, [])
        for names in [['one'], ['same'] * 1065, [None] * 1065]:
            with self.assertRaises(ValueError):
                objective(names=names)
        for kwargs in [
            dict(ties='efron'),
            dict(ties='native'),
            dict(ties='native', native_criterion=torch.mean),
            dict(native_criterion=torch.mean),
            dict(criterion_id='unused'),
            dict(
                ties='native',
                native_criterion=nn.Linear(1, 1),
                criterion_id='trainable',
            ),
        ]:
            with self.assertRaises(ValueError):
                objective(**kwargs)

    def test_endpoint_policy_metadata_safe_checkpoint_and_exact_optimizer_resume(self):
        torch.manual_seed(8)
        model = objective().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        x, demo = torch.randn(3, 2, 4, 128), torch.randn(3, 2)
        times = torch.tensor([3, 1, 2]).unsqueeze(1).expand(3, 1065)
        events = torch.ones(3, 1065, dtype=torch.bool)
        for _ in range(2):
            optimizer.zero_grad()
            model(x, demo, times, events)['loss'].backward()
            optimizer.step()
        stream = io.BytesIO()
        torch.save(
            dict(model=model.state_dict(), optimizer=optimizer.state_dict()), stream
        )
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = objective().train()
        restored.to_empty(device='cpu')
        restored.load_state_dict(saved['model'], strict=True)
        second = torch.optim.AdamW(restored.parameters(), lr=0.001)
        second.load_state_dict(saved['optimizer'])
        for net, opt in [(model, optimizer), (restored, second)]:
            opt.zero_grad()
            net(x, demo, times, events)['loss'].backward()
            opt.step()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(
                value, restored.state_dict()[key], rtol=0, atol=0
            )
        for a, b in zip(optimizer.state.values(), second.state.values()):
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        names = list(model.endpoint_names)
        names[0], names[1] = names[1], names[0]
        for incompatible in [
            objective(names=names),
            objective('native', lambda *x: x[0].sum(), 'other'),
            objective('native', lambda *x: x[0].sum(), 'new_source'),
        ]:
            with self.assertRaisesRegex(
                RuntimeError, 'endpoint order, tie policy or criterion identity'
            ):
                incompatible.load_state_dict(saved['model'], strict=True)
        state = copy.deepcopy(saved['model'])
        state['objective_fingerprint'][0] ^= 1
        with self.assertRaises(RuntimeError):
            objective().load_state_dict(state, strict=True)
        with torch.no_grad():
            model.objective_fingerprint[0] ^= 1
        with self.assertRaisesRegex(ValueError, 'matching SleepFM objective'):
            model(x, demo, times, events)


if __name__ == '__main__':
    unittest.main()
