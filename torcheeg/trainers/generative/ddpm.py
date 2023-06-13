import warnings
from contextlib import contextmanager
from functools import partial
from inspect import isfunction
from typing import Any, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torchmetrics.image.inception import InceptionScore

from .utils import FrechetInceptionDistance

_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader


def extract_into_tensor(a: torch.Tensor, t: int,
                        x_shape: Tuple[int, ...]) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1, ) * (len(x_shape) - 1)))


def noise_like(shape: Tuple[int, ...],
               device: torch.device,
               repeat: bool = False) -> torch.Tensor:
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1, ) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def betas_for_alpha_bar(num_diffusion_timesteps: int,
                        alpha_bar: Any,
                        max_beta: float = 0.999) -> np.ndarray:
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def make_beta_schedule(schedule: str,
                       n_timestep: int,
                       linear_start: float = 1e-4,
                       linear_end: float = 2e-2,
                       cosine_s: float = 8e-3) -> np.ndarray:
    if schedule == "linear":
        betas = (torch.linspace(linear_start**0.5,
                                linear_end**0.5,
                                n_timestep,
                                dtype=torch.float64)**2)

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep +
            cosine_s)
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start,
                               linear_end,
                               n_timestep,
                               dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start,
                               linear_end,
                               n_timestep,
                               dtype=torch.float64)**0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


class LitEma(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 decay: float = 0.9999,
                 use_num_upates: bool = True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer(
            'num_updates',
            torch.tensor(0, dtype=torch.int)
            if use_num_upates else torch.tensor(-1, dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                # remove as '.'-character is not allowed in buffers
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def reset_num_updates(self):
        del self.num_updates
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int))

    def forward(self, model: nn.Module):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay,
                        (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(
                        m_param[key])
                    shadow_params[sname].sub_(
                        one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model: nn.Module):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(
                    shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters: List[nn.Parameter]):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters: List[nn.Parameter]):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class DDPMTrainer(pl.LightningModule):
    '''
    This class implements the classic Denoising Diffusion Probabilistic Model (DDPM). The DDPM consists of two processes, the forward process, and the backward process. The forward process is to gradually add Gaussian noise to an image until it becomes random noise, while the backward process is the de-noising process. We train an attention-based UNet network at the backward process to start with random noise and gradually de-noise it until an image is generated and use the UNet to generate a simulated image from random noises. 

    - Paper: Ho J, Chen X, Srinivas A, et al. Denoising diffusion probabilistic models[J]. arXiv preprint arXiv:2006.11239, 2020.
    - URL: https://arxiv.org/abs/2006.11239
    - Related Project: https://github.com/Stability-AI/stablediffusion

    .. code-block:: python
        
        trainer = DDPMTrainer(model)
        trainer.fit(train_loader, val_loader, max_epochs=1)
        trainer.test(test_loader)

    Args:
        model (nn.Module): The denoising model takes the noisy samples and the current denoising conditions as input to predict the denoised samples. In this class, the current denoising condition is the current denoising time step. Typically, this model will be a UNet.
        lr (float): The learning rate. (default: :obj:`1e-4`)
        weight_decay (float): The weight decay. (default: :obj:`0.0`)
        timesteps (int): The number of steps in the diffusion process. (default: :obj:`1000`)
        beta_schedule (str): The schedule of the beta. Available options are: 'linear', 'cosine', 'sqrt_linear', 'sqrt'. (default: :obj:`"linear"`)
        loss_type (str): The type of the loss. Available options are: 'l2', 'l1'. (default: :obj:`"l2"`)
        use_ema (bool): Whether to use the exponential moving average. (default: :obj:`True`)
        clip_denoised (bool): Whether to clip the denoised image. (default: :obj:`True`)
        linear_start (float): The start value of the linear schedule. (default: :obj:`1e-4`)
        linear_end (float): The end value of the linear schedule. (default: :obj:`2e-2`)
        cosine_s (float): The cosine schedule. (default: :obj:`8e-3`)
        original_elbo_weight (float): The weight of the original ELBO loss. (default: :obj:`0.0`)
        v_posterior (float): The weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta. (default: :obj:`0.0`)
        l_simple_weight (float): The weight of the simple loss. (default: :obj:`1.0`)
        parameterization (str): The parameterization of the loss. Available options are: 'eps', 'x0', 'v'. (default: :obj:`"eps"`)
        use_positional_encodings (bool): Whether to use positional encodings. (default: :obj:`False`)
        learn_logvar (bool): Whether to learn the logvar. (default: :obj:`False`)
        logvar_init (float): The initial value of the logvar. (default: :obj:`0.0`)
        devices (int): The number of GPUs to use. (default: :obj:`1`)
        accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
        metrics (list of str): The metrics to use. Available options are: 'fid', 'is'. Due to the time-consuming generation process, these indicators will only be calculated and printed during test. (default: :obj:`[]`)
        metric_extractor (nn.Module): The feature extractor model for computing the FID score. (default: :obj:`None`)
        metric_classifier (nn.Module): The classifier model for computing the IS score. (default: :obj:`None`)
        metric_num_features (int): The number of features extracted by the metric_extractor. If not specified, it will be inferred from the :obj:`in_channels` attribute of the metric_extractor. (default: :obj:`None`)

    .. automethod:: fit
    .. automethod:: test
    '''
    def __init__(
            self,
            model: nn.Module,
            lr: float = 1e-4,
            weight_decay: float = 0.,
            timesteps: int = 1000,
            beta_schedule: str = "linear",
            loss_type: str = "l2",
            use_ema: bool = True,
            clip_denoised: bool = True,
            linear_start: float = 1e-4,
            linear_end: float = 2e-2,
            cosine_s: float = 8e-3,
            original_elbo_weight: float = 0.,
            v_posterior:
        float = 0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
            l_simple_weight: float = 1.,
            parameterization:
        str = "eps",  # all assuming fixed variance schedules
            learn_logvar: bool = False,
            logvar_init: float = 0.,
            devices: int = 1,
            accelerator: str = "cpu",
            metrics: List[str] = [],
            metric_extractor: nn.Module = None,
            metric_classifier: nn.Module = None,
            metric_num_features: int = None):
        super().__init__()
        assert parameterization in [
            "eps", "x0", "v"
        ], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        self.lr = lr
        self.weight_decay = weight_decay

        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.model = model
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        self.register_schedule(beta_schedule=beta_schedule,
                               timesteps=timesteps,
                               linear_start=linear_start,
                               linear_end=linear_end,
                               cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init,
                                 size=(self.num_timesteps, ))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self.metric_extractor = metric_extractor
        self.metric_classifier = metric_classifier
        self.metric_num_features = metric_num_features

        self.init_metrics(metrics)

    def init_metrics(self, metrics) -> None:
        self.train_simple_loss = torchmetrics.MeanMetric()
        self.val_simple_loss = torchmetrics.MeanMetric()
        self.test_simple_loss = torchmetrics.MeanMetric()

        self.train_vlb_loss = torchmetrics.MeanMetric()
        self.val_vlb_loss = torchmetrics.MeanMetric()
        self.test_vlb_loss = torchmetrics.MeanMetric()

        if 'fid' in metrics:
            assert not self.metric_extractor is None, 'The metric_extractor should be specified.'
            if hasattr(self.metric_extractor,
                       'in_channels') and self.metric_num_features is None:
                warnings.warn(
                    f'No metric_num_features specified, use metric_extractor.in_channels ({self.metric_extractor.in_channels}) as metric_num_features.'
                )
                self.metric_num_features = self.metric_extractor.in_channels
            assert not self.metric_num_features is None, 'The metric_num_features should be specified.'
            self.test_fid = FrechetInceptionDistance(self.metric_extractor,
                                                     self.metric_num_features)

        if 'is' in metrics:
            assert not self.metric_extractor is None, 'The metric_classifier should be specified.'
            self.test_is = InceptionScore(self.metric_classifier)

    def register_schedule(self,
                          beta_schedule: str = "linear",
                          timesteps: int = 1000,
                          linear_start: float = 1e-4,
                          linear_end: float = 2e-2,
                          cosine_s: float = 8e-3):
        betas = make_beta_schedule(beta_schedule,
                                   timesteps,
                                   linear_start=linear_start,
                                   linear_end=linear_end,
                                   cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[
            0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = (1 - self.v_posterior) * betas * (
            1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer('posterior_variance', to_torch(posterior_variance))

        self.register_buffer(
            'posterior_log_variance_clipped',
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer(
            'posterior_mean_coef1',
            to_torch(betas * np.sqrt(alphas_cumprod_prev) /
                     (1. - alphas_cumprod)))
        self.register_buffer(
            'posterior_mean_coef2',
            to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) /
                     (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (2 * self.posterior_variance *
                                            to_torch(alphas) *
                                            (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (
                2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(
                self.betas**2 /
                (2 * self.posterior_variance * to_torch(alphas) *
                 (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def q_mean_variance(
            self, x_start: torch.Tensor,
            t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) *
            x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t,
                                       x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t,
                                           x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(
            self, x_t: torch.Tensor, t: torch.Tensor,
            noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) *
            x_t - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t,
                                      x_t.shape) * noise)

    def predict_start_from_z_and_v(
            self, x_t: torch.Tensor, t: torch.Tensor,
            v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t,
                                x_t.shape) * v)

    def predict_eps_from_z_and_v(
            self, x_t: torch.Tensor, t: torch.Tensor,
            v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t,
                                x_t.shape) * x_t)

    def q_posterior(
            self, x_start: torch.Tensor, x_t: torch.Tensor,
            t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) *
            x_start +
            extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = extract_into_tensor(self.posterior_variance, t,
                                                 x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self,
                 x: torch.Tensor,
                 t: torch.Tensor,
                 clip_denoised: bool = True,
                 repeat_noise: bool = False) -> torch.Tensor:
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1, ) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 *
                                            model_log_variance).exp() * noise

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) *
            x_start + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t,
                                          x_start.shape) * noise)

    def get_v(self, x: torch.Tensor, noise: torch.Tensor,
              t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t,
                                x.shape) * x)

    def get_loss(self,
                 pred: torch.Tensor,
                 target: torch.Tensor,
                 mean: bool = True) -> torch.Tensor:
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target,
                                                    pred,
                                                    reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(
                f"Parameterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        loss_dict.update({'simple_loss': loss.mean()})
        simple_loss = loss.mean() * self.l_simple_weight

        vlb_loss = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({'vlb_loss': vlb_loss})

        loss = simple_loss + self.original_elbo_weight * vlb_loss

        return loss, loss_dict

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        outputs = noisy

        b = noisy.shape[0]
        device = noisy.device

        for i in reversed(range(0, self.num_timesteps)):
            outputs = self.p_sample(outputs,
                                    torch.full((b, ),
                                               i,
                                               device=device,
                                               dtype=torch.long),
                                    clip_denoised=self.clip_denoised)

        return outputs

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:

        x, _ = batch

        t = torch.randint(0,
                          self.num_timesteps, (x.shape[0], ),
                          device=self.device).long()
        loss, loss_dict = self.p_losses(x, t)

        self.log("train_simple_loss",
                 self.train_simple_loss(loss_dict['simple_loss']),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)
        self.log("train_vlb_loss",
                 self.train_vlb_loss(loss_dict['vlb_loss']),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)

        return loss

    def on_train_epoch_end(self) -> None:
        if self.use_ema:
            self.model_ema(self.model)

        self.log("train_simple_loss",
                 self.train_simple_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        self.log("train_vlb_loss",
                 self.train_vlb_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)

        str = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train_"):
                str += f"{key}: {value:.3f} "
        print(str + '\n')

        # reset the metrics
        self.train_simple_loss.reset()
        self.train_vlb_loss.reset()

    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        x, _ = batch

        t = torch.randint(0,
                          self.num_timesteps, (x.shape[0], ),
                          device=self.device).long()
        _, loss_dict = self.p_losses(x, t)

        self.val_simple_loss.update(loss_dict['simple_loss'])
        self.val_vlb_loss.update(loss_dict['vlb_loss'])

    def on_validation_epoch_end(self) -> None:
        self.log("val_simple_loss",
                 self.val_simple_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        self.log("val_vlb_loss",
                 self.val_vlb_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)

        # print the metrics
        str = "\n[VAL] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("val_"):
                str += f"{key}: {value:.3f} "
        print(str + '\n')

        # reset the metrics
        self.val_simple_loss.reset()
        self.val_vlb_loss.reset()

    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:

        x, _ = batch
        t = torch.randint(0,
                          self.num_timesteps, (x.shape[0], ),
                          device=self.device).long()
        _, loss_dict = self.p_losses(x, t)

        self.test_simple_loss.update(loss_dict['simple_loss'])
        self.test_vlb_loss.update(loss_dict['vlb_loss'])

        noise = torch.randn_like(x)
        gen_x = self(noise)

        if 'fid' in self.metrics:
            self.test_fid.update(x, real=True)
            self.test_fid.update(gen_x, real=False)

        if 'is' in self.metrics:
            self.test_is.update(gen_x)

    def on_test_epoch_end(self) -> None:
        self.log("test_simple_loss",
                 self.test_simple_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        self.log("test_vlb_loss",
                 self.test_vlb_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)

        if 'fid' in self.metrics:
            self.log("test_fid",
                     self.test_fid.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)
        if 'is' in self.metrics:
            self.log("test_is",
                     self.test_is.compute()[0],
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[TEST] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("test_"):
                str += f"{key}: {value:.3f} "
        print(str + '\n')

        # reset the metrics
        self.test_simple_loss.reset()
        self.test_vlb_loss.reset()

        if 'fid' in self.metrics:
            self.test_fid.reset()
        if 'is' in self.metrics:
            self.test_is.reset()

    def configure_optimizers(self):
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params,
                                lr=self.lr,
                                weight_decay=self.weight_decay)
        return opt

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            max_epochs: int = 300,
            *args,
            **kwargs) -> Any:
        r'''
        Args:
            train_loader (DataLoader): Iterable DataLoader for traversing the training data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            max_epochs (int): Maximum number of epochs to train the model. (default: :obj:`300`)
        '''
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             max_epochs=max_epochs,
                             inference_mode=False,
                             *args,
                             **kwargs)
        return trainer.fit(self, train_loader, val_loader)

    def test(self, test_loader: DataLoader, *args,
             **kwargs) -> _EVALUATE_OUTPUT:
        r'''
        Args:
            test_loader (DataLoader): Iterable DataLoader for traversing the test data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
        '''
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             inference_mode=False,
                             *args,
                             **kwargs)
        return trainer.test(self, test_loader)

    def predict_step(self,
                     batch: Tuple[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: int = 0) -> torch.Tensor:
        x, _ = batch
        noise = torch.randn_like(x)
        return self(noise)


class CDDPMTrainer(DDPMTrainer):
    '''
    This class implements the conditional Denoising Diffusion Probabilistic Model (DDPM). It introduces categories as conditions on top of DDPM, allowing to control the categories of generated samples. The DDPM consists of two processes, the forward process, and the backward process. The forward process is to gradually add Gaussian noise to an image until it becomes random noise, while the backward process is the de-noising process. We train an attention-based UNet network at the backward process to start with random noise and gradually de-noise it until an image is generated and use the UNet to generate a simulated image from random noises. 

    - Paper: Ho J, Chen X, Srinivas A, et al. Denoising diffusion probabilistic models[J]. arXiv preprint arXiv:2006.11239, 2020.
    - URL: https://arxiv.org/abs/2006.11239
    - Related Project: https://github.com/Stability-AI/stablediffusion

    .. code-block:: python
        
        model = UNet()
        trainer = CDDPMTrainer(model)
        trainer.fit(train_loader, val_loader, max_epochs=1)
        trainer.test(test_loader)

    Args:
        model (nn.Module): The denoising model takes the noisy samples and the current denoising conditions as input to predict the denoised samples. In this class, the current denoising condition is the current denoising time step. Typically, this model will be a UNet.
        lr (float): The learning rate. (default: :obj:`1e-4`)
        weight_decay (float): The weight decay. (default: :obj:`0.0`)
        timesteps (int): The number of steps in the diffusion process. (default: :obj:`1000`)
        beta_schedule (str): The schedule of the beta. Available options are: 'linear', 'cosine', 'sqrt_linear', 'sqrt'. (default: :obj:`"linear"`)
        loss_type (str): The type of the loss. Available options are: 'l2', 'l1'. (default: :obj:`"l2"`)
        use_ema (bool): Whether to use the exponential moving average. (default: :obj:`True`)
        clip_denoised (bool): Whether to clip the denoised image. (default: :obj:`True`)
        linear_start (float): The start value of the linear schedule. (default: :obj:`1e-4`)
        linear_end (float): The end value of the linear schedule. (default: :obj:`2e-2`)
        cosine_s (float): The cosine schedule. (default: :obj:`8e-3`)
        original_elbo_weight (float): The weight of the original ELBO loss. (default: :obj:`0.0`)
        v_posterior (float): The weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta. (default: :obj:`0.0`)
        l_simple_weight (float): The weight of the simple loss. (default: :obj:`1.0`)
        parameterization (str): The parameterization of the loss. Available options are: 'eps', 'x0', 'v'. (default: :obj:`"eps"`)
        use_positional_encodings (bool): Whether to use positional encodings. (default: :obj:`False`)
        learn_logvar (bool): Whether to learn the logvar. (default: :obj:`False`)
        logvar_init (float): The initial value of the logvar. (default: :obj:`0.0`)
        devices (int): The number of GPUs to use. (default: :obj:`1`)
        accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
        metrics (list of str): The metrics to use. Available options are: 'fid', 'is'. Due to the time-consuming generation process, these indicators will only be calculated and printed during test. (default: :obj:`[]`)
        metric_extractor (nn.Module): The feature extractor model for computing the FID score. (default: :obj:`None`)
        metric_classifier (nn.Module): The classifier model for computing the IS score. (default: :obj:`None`)
        metric_num_features (int): The number of features extracted by the metric_extractor. If not specified, it will be inferred from the :obj:`in_channels` attribute of the metric_extractor. (default: :obj:`None`)

    .. automethod:: fit
    .. automethod:: test
    '''
    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        noise: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t, y)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(
                f"Parameterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        loss_dict.update({'simple_loss': loss.mean()})
        simple_loss = loss.mean() * self.l_simple_weight

        vlb_loss = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({'vlb_loss': vlb_loss})

        loss = simple_loss + self.original_elbo_weight * vlb_loss

        return loss, loss_dict

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:

        x, y = batch

        t = torch.randint(0,
                          self.num_timesteps, (x.shape[0], ),
                          device=self.device).long()
        loss, loss_dict = self.p_losses(x, t, y)

        self.log("train_simple_loss",
                 self.train_simple_loss(loss_dict['simple_loss']),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)
        self.log("train_vlb_loss",
                 self.train_vlb_loss(loss_dict['vlb_loss']),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        x, y = batch

        t = torch.randint(0,
                          self.num_timesteps, (x.shape[0], ),
                          device=self.device).long()
        loss, loss_dict = self.p_losses(x, t, y)

        self.val_simple_loss.update(loss_dict['simple_loss'])
        self.val_vlb_loss.update(loss_dict['vlb_loss'])

        return loss

    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:

        x, y = batch
        t = torch.randint(0,
                          self.num_timesteps, (x.shape[0], ),
                          device=self.device).long()
        loss, loss_dict = self.p_losses(x, t, y)

        self.test_simple_loss.update(loss_dict['simple_loss'])
        self.test_vlb_loss.update(loss_dict['vlb_loss'])

        noise = torch.randn_like(x)
        gen_x = self(noise, y)

        if 'fid' in self.metrics:
            self.test_fid.update(x, real=True)
            self.test_fid.update(gen_x, real=False)

        if 'is' in self.metrics:
            self.test_is.update(gen_x)

        return loss

    def p_mean_variance(
            self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor,
            clip_denoised: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model_out = self.model(x, t, y)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self,
                 x: torch.Tensor,
                 t: torch.Tensor,
                 y: torch.Tensor,
                 clip_denoised: bool = True,
                 repeat_noise: bool = False) -> torch.Tensor:
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, y=y, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1, ) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 *
                                            model_log_variance).exp() * noise

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shape, device = x.shape, x.device
        b = shape[0]
        outputs = torch.randn(shape, device=device)
        for i in reversed(range(0, self.num_timesteps)):
            outputs = self.p_sample(x=outputs,
                                    t=torch.full((b, ),
                                                 i,
                                                 device=device,
                                                 dtype=torch.long),
                                    y=y,
                                    clip_denoised=self.clip_denoised)

        return outputs

    def predict_step(self,
                     batch: Tuple[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: int = 0) -> torch.Tensor:
        x, y = batch
        noise = torch.randn_like(x)
        return self(noise, y)