from copy import deepcopy

import torch
from torchmetrics.image.fid import \
    FrechetInceptionDistance as _FrechetInceptionDistance


def _compute_fid(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor,
                 sigma2: torch.Tensor) -> torch.Tensor:
    r"""Compute adjusted version of `Fid Score`_.

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples

    Returns:
        Scalar value of the distance between sets.
    """
    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

    return a + b - 2 * c


class FrechetInceptionDistance(_FrechetInceptionDistance):
    '''
    This class is used to calculate the Frechet Inception Distance (FID) metric. It is used to evaluate the quality of the generated EEG signals. For more details, please refer to the following information. Redefine the original implementation of FID in torchmetrics to support EEG signals.

    - Paper: Heusel M, Ramsauer H, Unterthiner T, et al. Gans trained by a two time-scale update rule converge to a local nash equilibrium[J]. Advances in neural information processing systems, 2017, 30.
    - URL: https://arxiv.org/abs/1706.08500
    - Project: https://github.com/Lightning-AI/torchmetrics/blob/f94d167319d397a1fa4aee593b99f9765a6dfa12/src/torchmetrics/image/fid.py#L183
    '''

    def __init__(
        self,
        inception,
        num_features,
        reset_real_features=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.inception = inception
        self.num_features = num_features
        if not isinstance(reset_real_features, bool):
            raise ValueError(
                "Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features
        mx_nb_feets = (num_features, num_features)

        self.add_state("real_features_sum",
                       torch.zeros(num_features).double(),
                       dist_reduce_fx="sum")
        self.add_state("real_features_cov_sum",
                       torch.zeros(mx_nb_feets).double(),
                       dist_reduce_fx="sum")
        self.add_state("real_features_num_samples",
                       torch.tensor(0).long(),
                       dist_reduce_fx="sum")

        self.add_state("fake_features_sum",
                       torch.zeros(num_features).double(),
                       dist_reduce_fx="sum")
        self.add_state("fake_features_cov_sum",
                       torch.zeros(mx_nb_feets).double(),
                       dist_reduce_fx="sum")
        self.add_state("fake_features_num_samples",
                       torch.tensor(0).long(),
                       dist_reduce_fx="sum")

    def update(self, imgs: torch.Tensor, real: bool) -> None:
        features = self.inception(imgs)
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += imgs.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += imgs.shape[0]

    def compute(self) -> torch.Tensor:
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError(
                "More than one sample is required for both the real and fake distributed to compute FID"
            )
        mean_real = (self.real_features_sum /
                     self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum /
                     self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t(
        ).mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t(
        ).mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(mean_real.squeeze(0), cov_real,
                            mean_fake.squeeze(0), cov_fake).to(self.orig_dtype)

    def reset(self) -> None:
        if not self.reset_real_features:
            real_features_sum = deepcopy(self.real_features_sum)
            real_features_cov_sum = deepcopy(self.real_features_cov_sum)
            real_features_num_samples = deepcopy(self.real_features_num_samples)
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_cov_sum = real_features_cov_sum
            self.real_features_num_samples = real_features_num_samples
        else:
            super().reset()