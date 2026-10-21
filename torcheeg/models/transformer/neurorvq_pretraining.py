"""Symmetric masked-token training with complete external NeuroRVQ models."""

import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from .neurorvq import NeuroRVQ
from .neurorvq_tokenizer import NeuroRVQTokenizer


class NeuroRVQMaskedTokenPretrainer(nn.Module):
    r"""Connect all 32 native prediction heads to a frozen complete tokenizer.

    Both complete external models must use identical electrode order,
    vocabulary and physical sampling rate. No source or weights are imported
    or downloaded here. The tokenizer stays frozen and in read-only evaluation.

    Supply an explicit boolean mask [B,C,P] with both partitions nonempty.
    Stream ``8 * branch + residual`` follows the original 32-head order.
    Cross entropy is averaged over all heads and selected patches separately
    for each partition; the two means are summed, irrespective of partition
    sizes. This is an explicit TorchEEG objective: the released author source
    supplies native logits and tokens, but no stage-two trainer or loss
    reduction. It is not a claim to reproduce an unpublished training loop.

    Args:
        backbone (NeuroRVQ): Complete external 32-head EEG-v1 backbone.
        tokenizer (NeuroRVQTokenizer): Complete matching pretrained tokenizer,
            configured with ``track_usage=False``.
    """

    _objective = b'neurorvq-32-head-patch-mean-per-partition-sum-v1'

    def __init__(self, backbone, tokenizer):
        super().__init__()
        if not isinstance(backbone, NeuroRVQ) or not isinstance(
            tokenizer, NeuroRVQTokenizer
        ):
            raise ValueError(
                'Require complete NeuroRVQ and NeuroRVQTokenizer adapters.'
            )
        if tokenizer.track_usage:
            raise ValueError(
                'The frozen teacher requires read-only evaluation (track_usage=False).'
            )
        for name in [
            'channel_names',
            'channel_vocab',
            'sampling_rate',
            'patch_size',
            'max_patches',
        ]:
            if getattr(backbone, name) != getattr(tokenizer, name):
                raise ValueError(
                    'Teacher and backbone physical metadata differs: ' + name
                )
        teacher_parameters = {id(parameter) for parameter in tokenizer.parameters()}
        if any(
            id(parameter) in teacher_parameters for parameter in backbone.parameters()
        ):
            raise ValueError(
                'Teacher and trainable backbone must not share parameters.'
            )
        self.backbone = backbone
        self.tokenizer = tokenizer.requires_grad_(False).eval()
        self.register_buffer(
            'objective_fingerprint',
            torch.tensor(
                list(hashlib.sha256(self._objective).digest()), dtype=torch.uint8
            ),
        )

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
        configured = torch.tensor(
            list(hashlib.sha256(self._objective).digest()), dtype=torch.uint8
        )
        value = state_dict.get(prefix + 'objective_fingerprint')
        if isinstance(value, torch.Tensor) and (
            value.dtype != configured.dtype or not torch.equal(value.cpu(), configured)
        ):
            error_msgs.append('NeuroRVQ masked-token objective reduction differs.')
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def train(self, mode=True):
        super().train(mode)
        self.tokenizer.eval()
        return self

    def forward(self, x, mask):
        r"""Return the symmetric loss, both logits, ordered targets and mask.

        Logits are [32,selected_patch_count,8192] and
        [32,complement_patch_count,8192]. Targets are [32,B,C,P]. The tokenizer
        has no input or parameter gradient and performs no EMA/statistic
        update. The backbone receives gradients through both native views.
        """
        logits, symmetric_logits = self.backbone.forward_masked(x, mask)
        self.tokenizer.eval()
        with torch.no_grad():
            targets = self.tokenizer.get_codebook_indices(x)
        expected = (32,) + tuple(mask.shape)
        if (
            not isinstance(targets, torch.Tensor)
            or targets.shape != expected
            or targets.dtype != torch.long
            or targets.device != x.device
            or (targets < 0).any()
            or (targets >= 8192).any()
        ):
            raise ValueError(
                'Require all 32 same-device native 8192-class target streams.'
            )
        flattened = targets.flatten(1)
        selected = flattened[:, mask.flatten()]
        complement = flattened[:, ~mask.flatten()]
        for values, labels in [(logits, selected), (symmetric_logits, complement)]:
            if not isinstance(values, torch.Tensor) or values.shape != labels.shape + (
                8192,
            ):
                raise ValueError('Require both complete native 32-head logit tensors.')
        masked_loss = F.cross_entropy(logits.reshape(-1, 8192), selected.reshape(-1))
        symmetric_loss = F.cross_entropy(
            symmetric_logits.reshape(-1, 8192), complement.reshape(-1)
        )
        return {
            'loss': masked_loss + symmetric_loss,
            'masked_loss': masked_loss,
            'symmetric_loss': symmetric_loss,
            'logits': logits,
            'symmetric_logits': symmetric_logits,
            'targets': targets,
            'mask': mask,
        }
