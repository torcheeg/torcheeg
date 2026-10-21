"""Complete reconstruction and contrastive objective for external Uni-DMFM."""

import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from .unidmfm import UniDMFM


class UniDMFMPretrainer(nn.Module):
    r"""Original masked time/frequency MSE and external native NT-Xent loss.

    Supply the complete external backbone and the original, explicitly
    constructed NTXentLoss module. Author source and weights are not bundled.
    Clean EEG and both independently augmented views are [B,C,T] at 200 Hz.
    The explicit binary mask [B,C,P] masks ONLY the first augmented view.
    Both reconstruction targets use the original clean EEG. The frequency
    target is log(abs(STFT)+1e-6), not log1p or an unwindowed FFT.

    Args:
        backbone (UniDMFM): Complete original model, including all three heads.
        contrastive_loss (nn.Module): Explicit native stateless NTXentLoss,
            with a positive finite ``temperature`` attribute.
        time_weight (float): Time reconstruction coefficient. (default: :obj:`1`)
        frequency_weight (float): Frequency coefficient. (default: :obj:`1`)
        contrastive_weight (float): Contrastive coefficient. (default: :obj:`1`)
        stft_window (Tensor, optional): Fixed CPU float32 window [64]. By
            default, snapshot a periodic CPU Hann64. The window is saved.
    """

    def __init__(
        self,
        backbone,
        contrastive_loss,
        time_weight=1.0,
        frequency_weight=1.0,
        contrastive_weight=1.0,
        stft_window=None,
    ):
        super().__init__()
        if not isinstance(backbone, UniDMFM) or not isinstance(
            contrastive_loss, nn.Module
        ):
            raise ValueError(
                'Require a complete UniDMFM and explicit native loss module.'
            )
        temperature = getattr(contrastive_loss, 'temperature', None)
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
            or list(contrastive_loss.parameters())
            or contrastive_loss.state_dict()
        ):
            raise ValueError(
                'Require stateless native NTXentLoss with positive finite temperature.'
            )
        weights = [time_weight, frequency_weight, contrastive_weight]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in weights
        ) or not any(weights):
            raise ValueError(
                'Require finite nonnegative coefficients, at least one positive.'
            )
        if stft_window is None:
            stft_window = torch.hann_window(
                64, periodic=True, dtype=torch.float32, device='cpu'
            )
        if (
            not isinstance(stft_window, torch.Tensor)
            or stft_window.shape != (64,)
            or stft_window.dtype != torch.float32
            or stft_window.device.type != 'cpu'
            or not torch.isfinite(stft_window).all()
        ):
            raise ValueError('Require a finite fixed CPU float32 STFT window[64].')
        self.backbone = backbone
        self.contrastive_loss = contrastive_loss
        self.time_weight, self.frequency_weight, self.contrastive_weight = map(
            float, weights
        )
        self._temperature = float(temperature)
        self._digest = hashlib.sha256(
            json.dumps(
                dict(
                    version=1,
                    weights=list(map(float, weights)),
                    temperature=self._temperature,
                    n_fft=64,
                    hop_length=16,
                    center=True,
                    pad_mode='reflect',
                    onesided=True,
                    target='log(abs(STFT)+1e-6)',
                    epsilon=1e-8,
                )
            ).encode()
        ).digest()
        self.register_buffer(
            'objective_fingerprint', torch.tensor(list(self._digest), dtype=torch.uint8)
        )
        self.register_buffer('stft_window', stft_window.detach().clone())

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
        configured = torch.tensor(list(self._digest), dtype=torch.uint8)
        if isinstance(value, torch.Tensor) and (
            value.dtype != configured.dtype or not torch.equal(value.cpu(), configured)
        ):
            error_msgs.append('Uni-DMFM objective, temperature or STFT policy differs.')
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x, first_view, second_view, mask):
        r"""Return weighted loss, three components, full predictions and targets.

        MSE is averaged over masked samples/bins only. Empty masks retain the
        native zero reconstruction losses (denominators have epsilon1e-8).
        A batch of one retains the native zero contrastive loss; use at least
        two examples for meaningful negatives. Only the frequency target is
        detached; time-target and augmented-view gradients follow the source.
        """
        patches = self.backbone._patches(x)
        for view in [first_view, second_view]:
            if (
                not isinstance(view, torch.Tensor)
                or view.shape != x.shape
                or view.device != x.device
                or view.dtype != x.dtype
            ):
                raise ValueError(
                    'Require two matching same-device/dtype augmented EEG views.'
                )
            self.backbone._patches(view)
        if (
            not isinstance(mask, torch.Tensor)
            or mask.shape != patches.shape[:3]
            or mask.device != x.device
            or not (mask.dtype == torch.bool or mask.is_floating_point())
            or ((mask != 0) & (mask != 1)).any()
        ):
            raise ValueError('Require a same-device binary mask[B,C,P].')
        if self.contrastive_loss.temperature != self._temperature:
            raise ValueError(
                'Native contrastive temperature changed after construction.'
            )
        mask = mask.to(dtype=torch.float32)
        masked = self.backbone._patches(first_view) * (1 - mask.unsqueeze(-1))
        output = self.backbone.forward_pretraining(masked.reshape_as(x))
        second_projection = self.backbone.forward_contrastive(second_view)
        time_prediction = output['time_prediction']
        frequency_prediction = output['frequency_prediction']
        time_loss = (
            F.mse_loss(time_prediction, patches, reduction='none') * mask.unsqueeze(-1)
        ).sum() / (mask.sum() * 200 + 1e-8)
        with torch.no_grad():
            target = torch.stft(
                patches.reshape(-1, 200),
                n_fft=64,
                hop_length=16,
                window=self.stft_window,
                return_complex=True,
            )
            frequency_target = torch.log(target.abs() + 1e-6).flatten(1)
        frequency_loss = (
            F.mse_loss(
                frequency_prediction.reshape(-1, 429),
                frequency_target,
                reduction='none',
            )
            * mask.reshape(-1, 1)
        ).sum() / (mask.sum() * 429 + 1e-8)
        contrastive_loss = self.contrastive_loss(
            output['contrastive_projection'], second_projection
        )
        if not isinstance(contrastive_loss, torch.Tensor) or contrastive_loss.ndim != 0:
            raise ValueError('Native contrastive loss must return a scalar tensor.')
        loss = (
            self.time_weight * time_loss
            + self.frequency_weight * frequency_loss
            + self.contrastive_weight * contrastive_loss
        )
        return dict(
            loss=loss,
            time_loss=time_loss,
            frequency_loss=frequency_loss,
            contrastive_loss=contrastive_loss,
            time_prediction=time_prediction,
            frequency_prediction=frequency_prediction,
            frequency_target=frequency_target,
            first_projection=output['contrastive_projection'],
            second_projection=second_projection,
            mask=mask,
        )
