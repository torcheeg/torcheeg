import math
from contextlib import nullcontext
from itertools import combinations

import torch
from torch import nn

__all__ = ['SleepDGAlignmentLoss']


class SleepDGAlignmentLoss(nn.Module):
    r'''Multi-level feature alignment for SleepDG.

    Align source-domain epoch means, unbiased covariances and mean sequence
    correlation matrices. This follows the authors' fixed implementation at
    revision ``01d209d4fda874933a277a88bccdca6e3835e884``. Epoch discrepancies
    are summed over unordered domain pairs, with elementwise means within
    each pair. Sequence discrepancies are summed over matrix elements and
    averaged over present domain pairs. These two normalizations differ.

    A domain represents a source dataset in the published cross-dataset
    protocol. Target-domain features must not enter this loss. Domain labels
    are integer codes from zero to ``num_domains - 1``. Missing domains are
    allowed, and fewer than two present domains produce a differentiable zero.

    For degenerate inputs, a domain with one epoch has zero covariance;
    constant epoch features have zero correlations. The correlation denominator
    is clamped to ``eps``. These finite extensions differ from the authors'
    undefined singleton/zero-variance cases. FP16 and BF16 statistics are
    computed in FP32 outside autocast; FP64 inputs retain FP64 precision.

    - Paper: Wang J, Zhao H, Liu S, et al. SleepDG: A Domain Generalization
      Approach for Out-of-Domain Sleep Staging. AAAI, 2024.
    - URL: https://arxiv.org/abs/2401.05363
    - Related Project: https://github.com/wjq-learning/SleepDG

    .. code-block:: python

        import torch
        from torcheeg.trainers import SleepDGAlignmentLoss

        features = torch.randn(8, 20, 512, requires_grad=True)
        domains = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        loss = SleepDGAlignmentLoss()(features, domains)
        loss.backward()

    Args:
        num_domains (int): Number of available source-domain codes. At least
            one is required. (default: :obj:`4`)
        eps (float): Positive lower bound for a product of centered feature
            norms in sequence correlations. (default: :obj:`1e-12`)
    '''

    def __init__(self, num_domains: int = 4, eps: float = 1e-12):
        super().__init__()
        if isinstance(num_domains, bool) or not isinstance(num_domains, int) or num_domains < 1:
            raise ValueError('num_domains must be a positive integer.')
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError('eps must be a finite positive number.')
        self.num_domains = num_domains
        self.eps = float(eps)
        self._pairs = tuple(combinations(range(num_domains), 2))

    def forward(self, features: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
        r'''Compute a scalar alignment loss for source sequences.

        Args:
            features (torch.Tensor): Floating features of shape
                ``[batch, sequence, hidden]`` with nonempty axes.
            domains (torch.Tensor): Integer source-domain codes of shape
                ``[batch]`` on the same device. Out-of-range codes raise a
                index error, including in compiled/exported execution.
        '''
        if not isinstance(features, torch.Tensor) or not isinstance(domains, torch.Tensor):
            raise TypeError('features and domains must be tensors.')
        if not features.is_floating_point() or features.ndim != 3:
            raise ValueError('features must be a real floating [batch, sequence, hidden] tensor.')
        if any(size < 1 for size in features.shape):
            raise ValueError('features must have nonempty batch, sequence and hidden axes.')
        if domains.ndim != 1 or domains.shape[0] != features.shape[0]:
            raise ValueError('domains must have shape [batch].')
        if domains.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError('domains must contain integer source-domain codes.')
        if domains.device != features.device:
            raise ValueError('features and domains must be on the same device.')
        domains = domains.to(dtype=torch.long)

        try:
            autocast_enabled = torch.is_autocast_enabled(features.device.type)
        except TypeError:
            # Older PyTorch releases expose separate CPU/CUDA queries.
            autocast_enabled = (torch.is_autocast_cpu_enabled()
                                if features.device.type == 'cpu'
                                else torch.is_autocast_enabled())
        context = (torch.autocast(device_type=features.device.type, enabled=False)
                   if autocast_enabled else nullcontext())
        with context:
            values = features.float() if features.dtype in (torch.float16, torch.bfloat16) else features
            # Route invalid codes to an out-of-bounds lookup. Unlike an async
            # assertion, index bounds remain checked by compiled CPU kernels.
            valid = (domains >= 0) & (domains < self.num_domains)
            indices = torch.where(valid, domains, self.num_domains)
            assignments = torch.eye(self.num_domains, dtype=values.dtype,
                                    device=values.device).index_select(0, indices)
            if self.num_domains == 1:
                # Retain the lookup in the graph even when no pairs exist.
                return values.sum() * 0. + (assignments - 1.).square().sum()
            centered = values - values.mean(dim=2, keepdim=True)
            norms = torch.linalg.vector_norm(centered, dim=2, keepdim=True)
            denominator = torch.bmm(norms, norms.transpose(1, 2)).clamp_min(self.eps)
            relations = torch.bmm(centered, centered.transpose(1, 2)) / denominator
            means, covariances, mean_relations, present = [], [], [], []
            for domain_id in range(self.num_domains):
                weights = assignments[:, domain_id]
                sequences = weights.sum()
                count = sequences * values.shape[1]
                mean = (values * weights[:, None, None]).sum(dim=(0, 1)) / count.clamp_min(1.)
                deviations = (values - mean) * weights[:, None, None]
                flat = deviations.reshape(-1, values.shape[2])
                covariance = flat.transpose(0, 1).matmul(flat) / (count - 1.).clamp_min(1.)
                mean_relation = (relations * weights[:, None, None]).sum(dim=0) / sequences.clamp_min(1.)
                means.append(mean)
                covariances.append(covariance)
                mean_relations.append(mean_relation)
                present.append(sequences > 0)
            # The zero keeps a gradient connection for a single configured domain.
            epoch_loss = values.sum() * 0.
            sequence_loss = values.sum() * 0.
            pair_count = values.new_zeros(())
            for left, right in self._pairs:
                active = (present[left] & present[right]).to(dtype=values.dtype)
                epoch_loss = epoch_loss + active * (
                    (means[left] - means[right]).square().mean()
                    + (covariances[left] - covariances[right]).square().mean())
                sequence_loss = sequence_loss + active * (
                    mean_relations[left] - mean_relations[right]).square().sum()
                pair_count = pair_count + active
            return epoch_loss + sequence_loss / pair_count.clamp_min(1.)
