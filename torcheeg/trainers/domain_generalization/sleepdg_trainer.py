import math
from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..classifier import ClassifierTrainer
from .sleepdg import SleepDGAlignmentLoss

__all__ = ['SleepDGTrainer']


def _positive(value, name, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero)):
        raise ValueError(f'{name} must be a finite {"nonnegative" if allow_zero else "positive"} number.')


def _loss_values(value):
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


class SleepDGTrainer(ClassifierTrainer):
    r'''Train SleepDG with classification, reconstruction and source alignment.

    Training batches contain ``(eeg, labels, domains)``: EEG has shape
    ``[batch, sequence, electrodes, samples]``, integer labels have shape
    ``[batch, sequence]``, and source-dataset codes have shape ``[batch]``.
    Source codes range from zero through ``num_domains - 1``. Keep target
    features out of training. Validation must use source-held-out data when
    selecting a model for an unseen target dataset.

    Validation/test batches contain ``(eeg, labels)`` or the same three-tuple;
    domain codes are ignored. These phases call encoder/classifier inference
    only and report epoch-level cross-entropy and classification metrics.
    Predict accepts EEG alone or either batch tuple and returns sequence logits.

    Default optimization follows the fixed author defaults: Adam at 1e-3,
    weight decay 1e-4, label smoothing 0.1 and unit loss coefficients. The fit
    wrapper defaults to 50 epochs and gradient-norm clipping at 1.0. The cosine
    scheduler steps after optimizer updates, with a horizon inferred from
    Lightning's estimated optimizer steps unless ``scheduler_steps`` is set.
    Explicitly set weight decay to ``lr / 10`` when following the author rule
    with a different learning rate. Model selection callbacks, data preparation
    and source sampling remain explicit workflow choices.

    - Paper: Wang J, Zhao H, Liu S, et al. SleepDG: A Domain Generalization
      Approach for Out-of-Domain Sleep Staging. AAAI, 2024.
    - URL: https://arxiv.org/abs/2401.05363
    - Related Project: https://github.com/wjq-learning/SleepDG

    Args:
        model (nn.Module): Sequence model with ``forward_all(eeg)`` returning
            logits, reconstruction and latent features, and ``forward(eeg)``
            returning logits without reconstruction.
        num_classes (int): Number of per-epoch classes. (default: :obj:`5`)
        num_domains (int): Available source-dataset codes. (default: :obj:`4`)
        lr (float): Adam learning rate. (default: :obj:`0.001`)
        weight_decay (float): Adam weight decay. (default: :obj:`0.0001`)
        weight_reconstruction (float): Nonnegative reconstruction coefficient.
            (default: :obj:`1.0`)
        weight_alignment (float): Nonnegative combined epoch/sequence alignment
            coefficient. (default: :obj:`1.0`)
        label_smoothing (float): Cross-entropy smoothing in all labeled phases,
            from zero through one. (default: :obj:`0.1`)
        lr_scheduler (bool): Enable a per-optimizer-step cosine schedule.
            (default: :obj:`True`)
        scheduler_steps (int, optional): Planned optimizer steps for the cosine
            horizon. Use the same horizon when resuming an interrupted run.
            None uses the attached Lightning trainer's estimate.
            (default: :obj:`None`)
        optimizer_eps (float): Adam numerical epsilon. (default: :obj:`1e-8`)
        devices (int): Devices passed to Lightning. (default: :obj:`1`)
        accelerator (str): Lightning accelerator. (default: :obj:`'cpu'`)
        verbose (bool): Use the base trainer's metric logging. Statistics are
            still updated when logging is disabled. (default: :obj:`True`)
        metrics (list of str): Base classification metrics, computed per epoch.
            (default: :obj:`['accuracy']`)

    .. automethod:: fit
    .. automethod:: test
    '''

    def __init__(self, model: nn.Module, num_classes: int = 5, num_domains: int = 4,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 weight_reconstruction: float = 1., weight_alignment: float = 1.,
                 label_smoothing: float = .1, lr_scheduler: bool = True,
                 scheduler_steps: Union[None, int] = None, optimizer_eps: float = 1e-8,
                 devices: int = 1, accelerator: str = 'cpu', verbose: bool = True,
                 metrics: List[str] = ['accuracy']):
        if not isinstance(model, nn.Module) or not callable(getattr(model, 'forward_all', None)):
            raise TypeError('model must be an nn.Module with forward_all.')
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError('num_classes must be an integer of at least two.')
        if hasattr(model, 'num_classes') and model.num_classes != num_classes:
            raise ValueError('model.num_classes must match the trainer.')
        for name, value, zero in [('lr', lr, False), ('weight_decay', weight_decay, True),
                                  ('weight_reconstruction', weight_reconstruction, True),
                                  ('weight_alignment', weight_alignment, True),
                                  ('optimizer_eps', optimizer_eps, False)]:
            _positive(value, name, zero)
        if (isinstance(label_smoothing, bool) or not isinstance(label_smoothing, (int, float))
                or not math.isfinite(label_smoothing) or not 0 <= label_smoothing <= 1):
            raise ValueError('label_smoothing must be between zero and one.')
        if not isinstance(lr_scheduler, bool):
            raise ValueError('lr_scheduler must be a boolean.')
        if scheduler_steps is not None and (isinstance(scheduler_steps, bool)
                or not isinstance(scheduler_steps, int) or scheduler_steps < 1):
            raise ValueError('scheduler_steps must be None or a positive integer.')
        alignment = SleepDGAlignmentLoss(num_domains)
        super().__init__(model=model, num_classes=num_classes, lr=lr,
                         weight_decay=weight_decay, devices=devices,
                         accelerator=accelerator, verbose=verbose, metrics=metrics)
        self.num_domains = num_domains
        self.weight_reconstruction = weight_reconstruction
        self.weight_alignment = weight_alignment
        self.label_smoothing = label_smoothing
        self.lr_scheduler = lr_scheduler
        self.scheduler_steps = scheduler_steps
        self.optimizer_eps = optimizer_eps
        self._optimizer_step_skipped = False
        self.alignment_fn = alignment
        self.ce_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.save_hyperparameters(ignore=['model'])

    def _batch(self, batch, training=False):
        expected = (3,) if training else (2, 3)
        if not isinstance(batch, (tuple, list)) or len(batch) not in expected:
            raise ValueError('Training requires (eeg, labels, domains); evaluation requires (eeg, labels).')
        eeg, labels = batch[:2]
        if (not isinstance(eeg, torch.Tensor) or not eeg.is_floating_point()
                or eeg.ndim != 4 or any(size < 1 for size in eeg.shape)):
            raise ValueError('eeg must be a nonempty floating [batch, sequence, electrodes, samples] tensor.')
        if (not isinstance(labels, torch.Tensor) or labels.shape != eeg.shape[:2]
                or labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                or labels.device != eeg.device):
            raise ValueError('labels must be integer [batch, sequence] targets on the EEG device.')
        labels = labels.long()
        if not ((labels >= 0) & (labels < self.num_classes)).all():
            raise ValueError('labels contain an out-of-range class.')
        if training:
            domains = batch[2]
            if (not isinstance(domains, torch.Tensor) or domains.shape != (eeg.shape[0],)
                    or domains.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                    or domains.device != eeg.device):
                raise ValueError('domains must be integer [batch] codes on the EEG device.')
            domains = domains.long()
            if not ((domains >= 0) & (domains < self.num_domains)).all():
                raise ValueError('domains contain an out-of-range source code.')
            return eeg, labels, domains
        return eeg, labels

    def _logits(self, logits, labels):
        if (not isinstance(logits, torch.Tensor) or not logits.is_floating_point()
                or logits.device != labels.device or logits.shape != (*labels.shape, self.num_classes)):
            raise ValueError('Model logits must have shape [batch, sequence, num_classes].')
        return _loss_values(logits).reshape(-1, self.num_classes), labels.reshape(-1)

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        eeg, labels, domains = self._batch(batch, training=True)
        outputs = self.model.forward_all(eeg)
        if (not isinstance(outputs, (tuple, list)) or len(outputs) != 3
                or not all(isinstance(value, torch.Tensor) and value.is_floating_point()
                           for value in outputs)):
            raise ValueError('forward_all must return floating logits, reconstruction and features.')
        logits, reconstruction, features = outputs
        if reconstruction.shape != eeg.shape:
            raise ValueError('Reconstruction must match EEG shape without broadcasting.')
        if features.ndim != 3 or features.shape[:2] != eeg.shape[:2]:
            raise ValueError('Features must have shape [batch, sequence, hidden].')
        flat_logits, flat_labels = self._logits(logits, labels)
        classification = self.ce_fn(flat_logits, flat_labels)
        reconstruction_loss = F.mse_loss(_loss_values(reconstruction), _loss_values(eeg))
        alignment = self.alignment_fn(features, domains)
        loss = classification + self.weight_reconstruction * reconstruction_loss + self.weight_alignment * alignment
        self.train_loss.update(loss.detach(), weight=flat_labels.numel())
        self.train_metrics.update(flat_logits.detach(), flat_labels)
        if self.verbose:
            self.log('train_loss', self.train_loss.compute(), prog_bar=True,
                     on_step=True, on_epoch=False, logger=False)
            for name, metric in zip(self.metrics, self.train_metrics.values()):
                self.log(f'train_{name}', metric.compute(), prog_bar=True,
                         on_step=True, on_epoch=False, logger=False)
        return loss

    def _evaluate(self, batch, phase):
        eeg, labels = self._batch(batch)
        logits, labels = self._logits(self(eeg), labels)
        loss = self.ce_fn(logits, labels)
        getattr(self, f'{phase}_loss').update(loss.detach(), weight=labels.numel())
        getattr(self, f'{phase}_metrics').update(logits.detach(), labels)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        return self._evaluate(batch, 'val')

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        return self._evaluate(batch, 'test')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, torch.Tensor):
            eeg = batch
        elif isinstance(batch, (tuple, list)) and len(batch) in (2, 3):
            eeg = batch[0]
        else:
            raise ValueError('Prediction requires EEG or (eeg, labels[, domains]).')
        if (not isinstance(eeg, torch.Tensor) or not eeg.is_floating_point()
                or eeg.ndim != 4 or any(size < 1 for size in eeg.shape)):
            raise ValueError('eeg must be a nonempty floating sequence tensor.')
        logits = self(eeg)
        if (not isinstance(logits, torch.Tensor) or not logits.is_floating_point()
                or logits.shape != (*eeg.shape[:2], self.num_classes)):
            raise ValueError('Model logits must have shape [batch, sequence, num_classes].')
        return logits

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, max_epochs: int = 50,
            *args, ckpt_path: Union[None, str] = None, **kwargs) -> Any:
        r'''Fit source batches, with default gradient-norm clipping at 1.0.

        ``ckpt_path`` resumes a trusted Lightning checkpoint. Other keyword
        arguments configure Lightning's Trainer. Use source validation and an
        explicit checkpoint callback to select a model before target testing.

        Args:
            train_loader (DataLoader): Labeled source sequences and dataset codes.
            val_loader (DataLoader): Source validation sequences and epoch labels.
            max_epochs (int): Maximum training epochs. (default: :obj:`50`)
            ckpt_path (str, optional): Trusted checkpoint to resume. (default: :obj:`None`)
        '''
        kwargs.setdefault('gradient_clip_val', 1.)
        trainer = pl.Trainer(devices=self.devices, accelerator=self.accelerator,
                             max_epochs=max_epochs, *args, **kwargs)
        return trainer.fit(self, train_loader, val_loader, ckpt_path=ckpt_path)

    def test(self, test_loader: DataLoader, *args, ckpt_path: Union[None, str] = None,
             **kwargs) -> List[Dict[str, float]]:
        r'''Evaluate sequence classification without reconstruction/alignment.

        Pass the actual selected checkpoint path to ``ckpt_path``; a new
        evaluator cannot infer the previous fit callback's best checkpoint.

        Args:
            test_loader (DataLoader): Evaluation sequences and epoch labels.
            ckpt_path (str, optional): Actual selected checkpoint path, or None
                for the current model. (default: :obj:`None`)
        '''
        trainer = pl.Trainer(devices=self.devices, accelerator=self.accelerator,
                             *args, **kwargs)
        return trainer.test(self, test_loader, ckpt_path=ckpt_path)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        scaler = getattr(self.trainer.precision_plugin, 'scaler', None)
        previous_scale = scaler.get_scale() if scaler is not None else None
        result = super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        # Lightning counts AMP attempts as steps even when GradScaler skips Adam.
        self._optimizer_step_skipped = (scaler is not None
                                        and scaler.get_scale() < previous_scale)
        return result

    def lr_scheduler_step(self, scheduler, metric):
        if not self._optimizer_step_skipped:
            super().lr_scheduler_step(scheduler, metric)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam((p for p in self.model.parameters() if p.requires_grad),
                                     lr=self.lr, weight_decay=self.weight_decay, eps=self.optimizer_eps)
        if not self.lr_scheduler:
            return optimizer
        steps = self.scheduler_steps
        if steps is None:
            steps = self.trainer.estimated_stepping_batches
        if not math.isfinite(steps) or steps < 1:
            raise ValueError('A finite optimizer-step estimate or scheduler_steps is required.')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(steps))
        return {'optimizer': optimizer, 'lr_scheduler': {
            'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}
