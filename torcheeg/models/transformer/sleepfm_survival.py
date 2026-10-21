"""Multi-endpoint survival objectives over complete external SleepFM models."""

import hashlib
import json

import torch
from torch import nn

from .sleepfm import SleepFMDiagnosis, _ordered_names


def _survival_targets(event_times, is_event, shape, device):
    if (
        not isinstance(event_times, torch.Tensor)
        or event_times.shape != shape
        or event_times.device != device
        or event_times.dtype == torch.bool
        or event_times.is_complex()
        or not torch.isfinite(event_times).all()
        or (event_times < 0).any()
    ):
        raise ValueError('Require finite nonnegative same-device times[B,1065].')
    if (
        not isinstance(is_event, torch.Tensor)
        or is_event.shape != shape
        or is_event.device != device
        or is_event.is_complex()
        or not torch.isfinite(is_event).all()
        or not ((is_event == 0) | (is_event == 1)).all()
    ):
        raise ValueError('Require same-device binary event indicators[B,1065].')


def _breslow_loss(log_risks, event_times, is_event):
    if log_risks.shape[0] == 1:
        # The only subject is its entire risk set: exact zero loss/gradient.
        return (log_risks * 0).sum()
    # No risk-set arithmetic is needed for an all-censored endpoint. This
    # also keeps extreme but finite, unused log risks out of subtractions.
    active = is_event.bool().any(dim=0)
    log_risks = torch.where(active.unsqueeze(0), log_risks, torch.zeros_like(log_risks))
    # Each endpoint has its own ordering and risk sets. Every tied row uses
    # the prefix ending at the LAST observation with that same event time.
    times, order = event_times.sort(dim=0, descending=True)
    risks = log_risks.gather(0, order)
    events = is_event.gather(0, order).to(risks.dtype)
    prefix = risks.logcumsumexp(dim=0)
    rows = torch.arange(risks.shape[0], device=risks.device).unsqueeze(1)
    ends = torch.cat(
        (times[:-1] != times[1:], torch.ones_like(times[-1:], dtype=torch.bool)),
        dim=0,
    )
    next_end = torch.where(ends, rows, risks.shape[0])
    next_end = next_end.flip(0).cummin(dim=0).values.flip(0)
    denominators = prefix.gather(0, next_end)
    # Zero-event endpoints contribute zero, but stay in the endpoint mean.
    per_endpoint = ((denominators - risks) * events).sum(0) / events.sum(0).clamp_min(1)
    return per_endpoint.mean()


class SleepFMSurvivalObjective(nn.Module):
    r"""Complete 1065-endpoint Cox training over an explicit SleepFM model.

    The default Breslow objective includes every subject whose observation
    time is greater than or equal to an event time in that event's risk set.
    This includes censored subjects at the same time. Tied event rows share
    the same denominator. Average event contributions within each endpoint,
    then average all 1065 endpoints, including zero-event endpoints.

    ``ties='native'`` instead calls an explicitly supplied original author
    criterion unchanged. The released criterion uses ordered prefix risk
    sets and float32 accumulation, and can depend on row order for tied
    times. It is not the default Breslow calculation. Source is not bundled.

    Args:
        model (SleepFMDiagnosis): Complete external diagnosis model.
        endpoint_names (list or tuple): All 1065 distinct ordered column names.
            Times and event indicators must use exactly this declared order.
        ties (str): ``'breslow'`` or explicit ``'native'``.
            (default: :obj:`'breslow'`)
        native_criterion (callable, optional): Explicit stateless author loss
            for native mode. Its signature is (log_risks, times, is_event).
        criterion_id (str, optional): Required native criterion identity, for
            example a pinned source revision and loss-source SHA256.

    This objective does not estimate baseline hazards or calibrated survival
    probabilities. A minibatch defines its own risk sets; it is not the full
    cohort likelihood unless the entire cohort is passed together. Supply
    measured follow-up times, censoring indicators and already encoded native
    demographics. A single subject or an all-censored endpoint has zero loss.
    """

    def __init__(
        self,
        model,
        endpoint_names,
        ties='breslow',
        native_criterion=None,
        criterion_id=None,
    ):
        super().__init__()
        if not isinstance(model, SleepFMDiagnosis):
            raise ValueError(
                'Require an explicitly constructed complete SleepFMDiagnosis.'
            )
        names = _ordered_names(endpoint_names)
        if len(names) != 1065 or any(not name.strip() for name in names):
            raise ValueError('Declare every one of the 1065 endpoint columns in order.')
        if ties not in ('breslow', 'native'):
            raise ValueError("Require ties='breslow' or explicit ties='native'.")
        if ties == 'native':
            if (
                not callable(native_criterion)
                or not isinstance(criterion_id, str)
                or not criterion_id.strip()
                or (
                    isinstance(native_criterion, nn.Module)
                    and (
                        list(native_criterion.parameters())
                        or native_criterion.state_dict()
                    )
                )
            ):
                raise ValueError(
                    'Native mode requires a stateless explicit criterion and identity.'
                )
        elif native_criterion is not None or criterion_id is not None:
            raise ValueError(
                'Breslow mode does not accept an unused native criterion or identity.'
            )
        self.model = model
        self.native_criterion = native_criterion
        self._endpoint_names, self._ties, self._criterion_id = names, ties, criterion_id
        self._digest = hashlib.sha256(
            json.dumps(
                dict(
                    version=1,
                    endpoint_names=names,
                    ties=ties,
                    criterion_id=criterion_id,
                    reduction='event_mean_then_all_endpoint_mean',
                    accumulation='native' if ties == 'native' else 'at_least_float32',
                )
            ).encode()
        ).digest()
        self.register_buffer(
            'objective_fingerprint', torch.tensor(list(self._digest), dtype=torch.uint8)
        )

    @property
    def endpoint_names(self):
        return self._endpoint_names

    @property
    def ties(self):
        return self._ties

    @property
    def criterion_id(self):
        return self._criterion_id

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        value = state_dict.get(prefix + 'objective_fingerprint')
        expected = torch.tensor(list(self._digest), dtype=torch.uint8)
        if isinstance(value, torch.Tensor) and (
            value.dtype != expected.dtype or not torch.equal(value.cpu(), expected)
        ):
            error_msgs.append(
                'SleepFM endpoint order, tie policy or criterion identity differs.'
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x, demographics, event_times, is_event, padding_mask=None):
        r"""Return loss and all native log-risk outputs, without a sigmoid.

        Embeddings are [B,modalities,steps,128], demographics [B,2], and both
        target tensors [B,1065]. Each endpoint's times and censoring convention
        must be consistent across subjects. Floating times are not quantized.
        Invalid targets are rejected before running or updating the model.
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[0] < 1:
            raise ValueError('Require embeddings[B,modalities,steps,128].')
        _survival_targets(event_times, is_event, (x.shape[0], 1065), x.device)
        expected = torch.tensor(list(self._digest), dtype=torch.uint8)
        if not torch.equal(self.objective_fingerprint.cpu(), expected):
            raise ValueError(
                'Restore a matching SleepFM objective state before training.'
            )
        log_risks = self.model(x, demographics, padding_mask)
        if not torch.isfinite(log_risks).all():
            raise ValueError('Require finite log-risk outputs from the complete model.')
        if self.ties == 'native':
            loss = self.native_criterion(log_risks, event_times, is_event)
        else:
            accumulated = (
                log_risks if log_risks.dtype == torch.float64 else log_risks.float()
            )
            loss = _breslow_loss(accumulated, event_times, is_event)
        if (
            not isinstance(loss, torch.Tensor)
            or loss.ndim != 0
            or not loss.is_floating_point()
            or loss.device != x.device
            or not torch.isfinite(loss)
        ):
            raise ValueError('Require a finite same-device scalar survival loss.')
        return dict(loss=loss, log_risks=log_risks)
