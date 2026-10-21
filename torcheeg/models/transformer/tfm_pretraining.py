# TFM pretraining adaptation; see LICENSES/TFMTokenizer.txt and BIOT.txt.
# Original source: Jathurshan0330/TFM-Tokenizer, revision 2d6da482b16dabbb2ebec808fa9f505fc6f367c4.
import torch
from torch import nn
from torch.nn import functional as F

from .tfm_tokenizer import TFMEncoder, TFMTokenizer


def _spectrum(eeg):
    if (
        not isinstance(eeg, torch.Tensor)
        or not eeg.is_floating_point()
        or eeg.ndim != 3
        or min(eeg.shape) < 1
        or eeg.shape[-1] < 200
    ):
        raise ValueError(
            'Require nonempty real floating EEG [batch,electrodes,samples>=200] at 200 Hz.'
        )
    temporal = eeg.reshape(-1, eeg.shape[-1])
    spectrum = torch.stft(
        temporal,
        n_fft=200,
        hop_length=100,
        window=torch.hann_window(200, dtype=torch.float32, device='cpu').to(eeg.device),
        center=False,
        onesided=True,
        return_complex=True,
    ).abs()[:, :100]
    return spectrum, temporal


class TFMTokenizerPretrainer(nn.Module):
    r"""Complete native complementary-spectrum TFM tokenizer training objective.

    Input is preprocessed floating EEG [batch,electrodes,samples] at 200 Hz.
    The full tokenizer runs twice on complementary masked spectra with the
    SAME UNMASKED waveform. Both reconstruction losses cover the entire clean
    spectrum. Training updates EMA twice, sequentially, as in the source.

    Native masking chooses half of twenty five-bin frequency groups and half
    the frames, rounding down. Their keep-region intersection is shared by
    every sample/electrode; the second view is its full complement. This is
    not two independent 50-percent masks. A torch.Generator for this mask
    must be on CPU; neural dropout still uses the ordinary device RNG.

    Args:
        tokenizer (TFMTokenizer): Complete original tokenizer and reconstruction head.
        quantization_mode (str): native retains the original codebook plus
            beta-weighted commitment terms, whose encoder gradients cancel
            at beta=1 with straight-through codes. ema explicitly uses the
            corrected detached-target commitment helper instead. Save this
            choice with the experiment. (default: :obj:`'native'`)
    """

    def __init__(self, tokenizer, quantization_mode='native'):
        super().__init__()
        if not isinstance(tokenizer, TFMTokenizer):
            raise ValueError('Require a complete TFMTokenizer.')
        if not isinstance(quantization_mode, str) or quantization_mode not in (
            'native',
            'ema',
        ):
            raise ValueError('quantization_mode must be native or ema.')
        self.tokenizer = tokenizer
        self.quantization_mode = quantization_mode

    @staticmethod
    def mask_spectrum(spectrum, generator=None):
        r"""Return the native batch-shared Boolean keep mask [batch,100,frames]."""
        if (
            not isinstance(spectrum, torch.Tensor)
            or spectrum.ndim != 3
            or not spectrum.is_floating_point()
            or min(spectrum.shape) < 1
            or spectrum.shape[1] != 100
        ):
            raise ValueError('Require real floating spectrum [batch,100,frames].')
        if generator is not None and (
            not isinstance(generator, torch.Generator) or generator.device.type != 'cpu'
        ):
            raise ValueError('Spectral masking requires a CPU torch.Generator.')
        frames = spectrum.shape[-1]
        frequency = torch.ones(20, dtype=torch.bool, device='cpu')
        frequency[torch.randperm(20, generator=generator, device='cpu')[:10]] = False
        temporal = torch.ones(frames, dtype=torch.bool, device='cpu')
        temporal[
            torch.randperm(frames, generator=generator, device='cpu')[: frames // 2]
        ] = False
        keep = frequency.repeat_interleave(5)[:, None] & temporal[None, :]
        return (
            keep.to(device=spectrum.device)
            .unsqueeze(0)
            .expand(spectrum.shape[0], -1, -1)
        )

    def forward(self, eeg, generator=None, mask=None):
        spectrum, temporal = _spectrum(eeg)
        if mask is None:
            mask = self.mask_spectrum(spectrum, generator)
        elif (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.shape != spectrum.shape
            or mask.device != spectrum.device
        ):
            raise ValueError(
                'Explicit mask must be Boolean and match the flattened spectrum shape/device.'
            )
        elif generator is not None:
            raise ValueError(
                'Supply an explicit mask or a masking generator, not both.'
            )
        first = self.tokenizer(spectrum * mask, temporal)
        second = self.tokenizer(spectrum * ~mask, temporal)
        reconstruction = (first[0] - spectrum).square().mean() + (
            second[0] - spectrum
        ).square().mean()
        if self.quantization_mode == 'native':

            def quantization(output):
                return (
                    output[2] - output[3].detach()
                ).square().mean() + self.tokenizer.beta * (
                    output[2].detach() - output[3]
                ).square().mean()
        else:

            def quantization(output):
                return self.tokenizer.quantization_loss(output[3], output[2])

        quantization_loss = quantization(first) + quantization(second)
        return dict(
            loss=reconstruction + quantization_loss,
            reconstruction_loss=reconstruction,
            quantization_loss=quantization_loss,
            reconstruction=first[0],
            complementary_reconstruction=second[0],
            ids=first[1],
            complementary_ids=second[1],
            mask=mask,
        )


class TFMMaskedTokenPretrainer(nn.Module):
    r"""Symmetric masked-token prediction with a frozen complete tokenizer teacher.

    Clean EEG [batch,1--16 electrodes,samples] at 200 Hz produces teacher
    IDs. The encoder must retain a genuine full-vocabulary output head,
    such as the multi-dataset MTP checkpoint. Each token is independently
    visible or masked in the first view and the opposite in the second.
    Two separately averaged masked-only cross-entropies are SUMMED, without
    partition-size weighting. Both partitions must be nonempty; reject the
    source's otherwise undefined empty-target CE instead of returning NaN.

    A supplied generator uses the token device. Explicit Boolean mask uses
    True=visible in the first view, in [batch,electrodes,frames] order.
    Hard IDs stop raw-input and teacher gradients. Teacher eval is maintained
    across train calls, so no_grad never silently updates its EMA state.

    Args:
        tokenizer (TFMTokenizer): Complete tokenizer, permanently frozen by this wrapper.
        encoder (TFMEncoder): Complete encoder with num_classes == num_tokens.
    """

    def __init__(self, tokenizer, encoder):
        super().__init__()
        if not isinstance(tokenizer, TFMTokenizer) or not isinstance(
            encoder, TFMEncoder
        ):
            raise ValueError('Require complete TFMTokenizer and TFMEncoder models.')
        if (
            tokenizer.num_tokens != encoder.num_tokens
            or tokenizer.hid_channels != encoder.hid_channels
            or encoder.num_classes != encoder.num_tokens
        ):
            raise ValueError(
                'Require matching codebooks/widths and a complete vocabulary output head.'
            )
        self.tokenizer = tokenizer.requires_grad_(False).eval()
        self.encoder = encoder

    def train(self, mode=True):
        super().train(mode)
        self.tokenizer.eval()
        return self

    @torch.no_grad()
    def get_codebook_indices(self, eeg):
        spectrum, temporal = _spectrum(eeg)
        if not 1 <= eeg.shape[1] <= 16 or spectrum.shape[-1] > 1000:
            raise ValueError(
                'Require 1--16 electrodes and at most 1000 temporal positions.'
            )
        _, ids, _ = self.tokenizer.tokenize(spectrum, temporal)
        return ids.reshape(eeg.shape[0], eeg.shape[1], -1)

    def forward(self, eeg, channel_ids=None, generator=None, mask=None):
        ids = self.get_codebook_indices(eeg)
        if ids.numel() < 2:
            raise ValueError('Symmetric prediction needs at least two token positions.')
        if mask is None:
            mask = torch.randint(
                0, 2, ids.shape, device=ids.device, generator=generator
            ).bool()
        elif (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.shape != ids.shape
            or mask.device != ids.device
        ):
            raise ValueError(
                'Explicit token mask must be Boolean with the teacher ID shape/device.'
            )
        elif generator is not None:
            raise ValueError(
                'Supply an explicit mask or a masking generator, not both.'
            )
        if not mask.any() or mask.all():
            raise ValueError(
                'Both symmetric mask partitions must be nonempty; supply another mask or advance the generator.'
            )
        first_ids = torch.where(mask, ids, self.encoder.num_tokens)
        second_ids = torch.where(~mask, ids, self.encoder.num_tokens)
        first = self.encoder.masked_prediction(first_ids, channel_ids)
        second = self.encoder.masked_prediction(second_ids, channel_ids)
        first_loss = F.cross_entropy(
            first.reshape(-1, self.encoder.num_tokens)[~mask.flatten()], ids[~mask]
        )
        second_loss = F.cross_entropy(
            second.reshape(-1, self.encoder.num_tokens)[mask.flatten()], ids[mask]
        )
        return dict(
            loss=first_loss + second_loss,
            first_loss=first_loss,
            complementary_loss=second_loss,
            prediction=first,
            complementary_prediction=second,
            ids=ids,
            mask=mask,
        )
