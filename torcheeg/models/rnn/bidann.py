# Adapted from XJTU-EEG/LibEER at dddff9776dbdae21195fe320dff0a5ba61628a18.
# Copyright (c) 2024 xjtu-eeg. MIT license; see LICENSES/BIDANN-LIBEER-LICENSE.txt.
# Paper: Li et al., doi:10.1109/TAFFC.2018.2885474.

import math
import torch
from torch import nn

_SEED_CHANNELS = [
    'FP1',
    'FPZ',
    'FP2',
    'AF3',
    'AF4',
    'F7',
    'F5',
    'F3',
    'F1',
    'FZ',
    'F2',
    'F4',
    'F6',
    'F8',
    'FT7',
    'FC5',
    'FC3',
    'FC1',
    'FCZ',
    'FC2',
    'FC4',
    'FC6',
    'FT8',
    'T7',
    'C5',
    'C3',
    'C1',
    'CZ',
    'C2',
    'C4',
    'C6',
    'T8',
    'TP7',
    'CP5',
    'CP3',
    'CP1',
    'CPZ',
    'CP2',
    'CP4',
    'CP6',
    'TP8',
    'P7',
    'P5',
    'P3',
    'P1',
    'PZ',
    'P2',
    'P4',
    'P6',
    'P8',
    'PO7',
    'PO5',
    'PO3',
    'POZ',
    'PO4',
    'PO6',
    'PO8',
    'CB1',
    'O1',
    'OZ',
    'O2',
    'CB2',
]

_LEFT_CHANNELS = [
    'FP1',
    'AF3',
    'F7',
    'F5',
    'F3',
    'F1',
    'FT7',
    'FC5',
    'FC3',
    'FC1',
    'T7',
    'C5',
    'C3',
    'C1',
    'TP7',
    'CP5',
    'CP3',
    'CP1',
    'P7',
    'P5',
    'P3',
    'P1',
    'PO7',
    'PO5',
    'PO3',
    'CB1',
    'O1',
    'FPZ',
    'FCZ',
    'CPZ',
    'POZ',
]

_RIGHT_CHANNELS = [
    'FP2',
    'AF4',
    'F8',
    'F6',
    'F4',
    'F2',
    'FT8',
    'FC6',
    'FC4',
    'FC2',
    'T8',
    'C6',
    'C4',
    'C2',
    'TP8',
    'CP6',
    'CP4',
    'CP2',
    'P8',
    'P6',
    'P4',
    'P2',
    'PO8',
    'PO6',
    'PO4',
    'CB2',
    'O2',
    'FZ',
    'CZ',
    'PZ',
    'OZ',
]


class BiDANN(nn.Module):
    r"""Complete LibEER bi-hemisphere domain-adversarial network.

    Two full two-layer LSTMs and temporal projections feed the categorical
    classifier, left/right local discriminators and global discriminator.
    Source and target input are floating
    :obj:`[batch, sequence_length, 62, in_channels]` SEED-order features.
    Calling with only source returns classification logits; providing an
    unlabeled target returns the native four-head tuple for adversarial loss.

    This is the MIT LibEER independent reimplementation, not a numerical
    equivalence claim for unavailable original-author code. Every native
    three-class state key and neural calculation is retained. Tensor-only
    hemisphere selection extends gradients to input and avoids CPU/NumPy
    detachment. Correct aligned frames are the default; the literal reference
    right reshape remains available for checkpoint replay.

    Args:
        num_electrodes (int): Exactly 62 SEED-ordered rows. (default: 62)
        in_channels (int): Features per electrode/frame. (default: 5)
        num_classes (int): Categorical output classes. The reference ignored
            this argument and fixed three; it is honored here. (default: 3)
        sequence_length (int): Frames per sequence. (default: 9)
        adversarial_weight (float): Nonnegative gradient-reversal scale,
            applied to encoder gradients from each discriminator. (default: 1.0)
        hemisphere_layout (str): Aligned channels/features within each time
            frame, or the reference's legacy right reshape. Use 'libeer' for
            strict legacy replay. (default: 'aligned')

    .. code-block:: python

        model = BiDANN()
        source = torch.randn(3, 9, 62, 5)
        target = torch.randn(2, 9, 62, 5)
        outputs = model(source, target)
        loss = BiDANNLoss()(outputs, torch.tensor([0, 1, 2]))
        loss.backward()
    """

    def __init__(
        self,
        num_electrodes: int = 62,
        in_channels: int = 5,
        num_classes: int = 3,
        sequence_length: int = 9,
        adversarial_weight: float = 1.0,
        hemisphere_layout: str = 'aligned',
    ):
        super(BiDANN, self).__init__()
        for name, value in [
            ('num_electrodes', num_electrodes),
            ('in_channels', in_channels),
            ('num_classes', num_classes),
            ('sequence_length', sequence_length),
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if num_electrodes != 62:
            raise ValueError(
                'The native SEED hemisphere mapping requires 62 ordered rows.'
            )
        if (
            isinstance(adversarial_weight, bool)
            or not isinstance(adversarial_weight, (int, float))
            or not math.isfinite(adversarial_weight)
            or adversarial_weight < 0
        ):
            raise ValueError('adversarial_weight must be finite and nonnegative.')
        if hemisphere_layout not in ('aligned', 'libeer'):
            raise ValueError("hemisphere_layout must be 'aligned' or 'libeer'.")
        self.hemisphere_layout = hemisphere_layout
        domain_classes = 2
        lambda_ = float(adversarial_weight)
        sample_length = sequence_length
        device = None
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.num_classes = num_classes
        self.domain_classes = domain_classes
        self.lambda_ = lambda_
        self.length = sample_length
        self.sequence_length = sequence_length
        self.device = device
        self.l_extractor = _BiDANNFeatureExtractor(
            num_electrodes=self.num_electrodes,
            in_channels=self.in_channels,
            sample_length=self.length,
            hidden_size=128,
            output_size=8,
        )
        self.r_extractor = _BiDANNFeatureExtractor(
            num_electrodes=self.num_electrodes,
            in_channels=self.in_channels,
            sample_length=self.length,
            hidden_size=128,
            output_size=8,
        )
        self.classifer = _BiDANNClassifier(
            input_size=8 * 128,
            hidden_size1=256,
            hideen_size2=64,
            num_classes=num_classes,
        )
        self.r_discriminator = _LocalDiscriminator(
            input_size=8 * 128,
            hidden_size1=256,
            hidden_size2=64,
            domain_classes=self.domain_classes,
            lambda_=self.lambda_,
        )
        self.l_discriminator = _LocalDiscriminator(
            input_size=8 * 128,
            hidden_size1=256,
            hidden_size2=64,
            domain_classes=self.domain_classes,
            lambda_=self.lambda_,
        )
        self.global_discriminator = _GlobalDiscriminator(
            input_size=8 * 128,
            hidden_size1=256,
            hidden_size2=64,
            domain_classes=self.domain_classes,
            lambda_=self.lambda_,
        )

    def forward(self, source_data: torch.Tensor, target_data=None):
        self._validate_input(source_data)
        if target_data is None:
            left, right = _split_hemispheres(
                source_data,
                _SEED_CHANNELS,
                _LEFT_CHANNELS,
                _RIGHT_CHANNELS,
                self.hemisphere_layout,
            )
            return self.classifer(self.l_extractor(left), self.r_extractor(right))
        self._validate_input(target_data)
        l_source_data, r_source_data = _split_hemispheres(
            source_data,
            columns=_SEED_CHANNELS,
            l_columns=_LEFT_CHANNELS,
            r_columns=_RIGHT_CHANNELS,
            layout=self.hemisphere_layout,
        )
        l_target_data, r_target_data = _split_hemispheres(
            target_data,
            columns=_SEED_CHANNELS,
            l_columns=_LEFT_CHANNELS,
            r_columns=_RIGHT_CHANNELS,
            layout=self.hemisphere_layout,
        )
        l_source_data = l_source_data.to(self.device)
        r_source_data = r_source_data.to(self.device)
        l_target_data = l_target_data.to(self.device)
        r_target_data = r_target_data.to(self.device)
        l_source_feature = self.l_extractor(l_source_data)
        r_source_feature = self.r_extractor(r_source_data)
        l_target_feature = self.l_extractor(l_target_data)
        r_target_feature = self.r_extractor(r_target_data)
        label_predict = self.classifer(l_source_feature, r_source_feature)
        l_domain_predict = self.l_discriminator(l_source_feature, l_target_feature)
        r_domain_predict = self.r_discriminator(r_source_feature, r_target_feature)
        global_domain_predict = self.global_discriminator(
            l_source_feature, r_source_feature, l_target_feature, r_target_feature
        )
        return (
            label_predict,
            l_domain_predict,
            r_domain_predict,
            global_domain_predict,
        )

    def _validate_input(self, x):
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError('Expected floating grouped EEG feature sequences.')
        if x.ndim != 4 or x.shape[1:] != (
            self.length,
            self.num_electrodes,
            self.in_channels,
        ):
            raise ValueError(
                'Expected [batch,sequence_length,62,in_channels] features in SEED order.'
            )

    @property
    def classifier(self):
        """Expose the full head while preserving native classifer state keys."""
        return self.classifer


def _split_hemispheres(features, columns, l_columns, r_columns, layout='libeer'):
    # Retain the reference's literal left/right layouts using differentiable indexing.
    values = features.transpose(2, 3)
    left_indices = torch.tensor(
        [columns.index(name) for name in l_columns], device=features.device
    )
    right_indices = torch.tensor(
        [columns.index(name) for name in r_columns], device=features.device
    )
    left = values.index_select(-1, left_indices).transpose(2, 3)
    left = left.reshape(features.shape[0], features.shape[1], -1)
    right = values.index_select(-1, right_indices)
    if layout == 'aligned':
        right = right.transpose(2, 3)
    else:
        right = right.transpose(1, 2)
    right = right.reshape(features.shape[0], features.shape[1], -1)
    return left, right


class _BiDANNFeatureExtractor(nn.Module):
    def __init__(
        self,
        num_electrodes=62,
        in_channels=5,
        sample_length=9,
        hidden_size=128,
        output_size=8,
    ):
        super(_BiDANNFeatureExtractor, self).__init__()
        self.input = num_electrodes * in_channels // 2
        self.out_lstm = hidden_size
        self.length = sample_length
        self.output = output_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=self.input,
            hidden_size=self.out_lstm,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )
        self.linear = nn.Linear(in_features=self.length, out_features=self.output)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)
        self.initialize()

    def initialize(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        x = x.reshape(-1, self.output * self.out_lstm)
        return x


class _BiDANNClassifier(nn.Module):
    def __init__(
        self, input_size=3 * 128, hidden_size1=128, hideen_size2=32, num_classes=3
    ):
        super(_BiDANNClassifier, self).__init__()
        self.input = input_size
        self.hidden1 = hidden_size1
        self.hidden2 = hideen_size2
        self.num_classes = num_classes
        self.classifer = nn.Sequential(
            nn.Linear(in_features=2 * self.input, out_features=2 * self.hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(2 * self.hidden1),
            nn.Linear(in_features=2 * self.hidden1, out_features=2 * self.hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(2 * self.hidden2),
            nn.Linear(in_features=2 * self.hidden2, out_features=self.num_classes),
        )
        self.initialize()

    def initialize(self):
        for m in self.classifer.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, l_feature, r_feature):
        source_feature = torch.cat((l_feature, r_feature), dim=1)
        source_feature = self.classifer(source_feature)
        return source_feature


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return (-ctx.lambda_ * grad_out, None)


class _LocalDiscriminator(nn.Module):
    def __init__(
        self,
        input_size=3 * 128,
        hidden_size1=128,
        hidden_size2=32,
        domain_classes=2,
        lambda_=0.5,
    ):
        super(_LocalDiscriminator, self).__init__()
        self.lambda_ = lambda_
        self.input_size = input_size
        self.hidden1 = hidden_size1
        self.hidden2 = hidden_size2
        self.domain_classes = domain_classes
        self.discriminator = nn.Sequential(
            nn.Linear(self.input_size, self.hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden1),
            nn.Linear(self.hidden1, self.hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden2),
            nn.Linear(self.hidden2, self.domain_classes),
        )
        self.initialize()

    def initialize(self):
        for m in self.discriminator:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, feature1, feature2):
        features = torch.cat((feature1, feature2), dim=0)
        reversed_features = _GradientReverse.apply(features, self.lambda_)
        output = self.discriminator(reversed_features)
        return output


class _GlobalDiscriminator(nn.Module):
    def __init__(
        self,
        input_size=3 * 128,
        hidden_size1=128,
        hidden_size2=32,
        domain_classes=2,
        lambda_=0.5,
    ):
        super(_GlobalDiscriminator, self).__init__()
        self.lambda_ = lambda_
        self.input_size = input_size
        self.hidden1 = hidden_size1
        self.hidden2 = hidden_size2
        self.output = domain_classes
        self.discriminator = nn.Sequential(
            nn.Linear(2 * self.input_size, 2 * self.hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(2 * self.hidden1),
            nn.Linear(2 * self.hidden1, 2 * self.hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(2 * self.hidden2),
            nn.Linear(2 * self.hidden2, self.output),
        )
        self.initialize()

    def initialize(self):
        for m in self.discriminator:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, l_source, r_source, l_target, r_target):
        source_features = torch.cat((l_source, r_source), dim=1)
        target_features = torch.cat((l_target, r_target), dim=1)
        all_features = torch.cat((source_features, target_features), dim=0)
        reversed_features = _GradientReverse.apply(all_features, self.lambda_)
        output = self.discriminator(reversed_features)
        return output


class BiDANNLoss(nn.Module):
    r"""Source cross entropy plus the three native domain cross entropies.

    Domain labels concatenate source zeros and target ones in native order.
    No target emotion label is accepted or used. Gradient reversal belongs
    to the model; this loss does not reverse gradients a second time.

    Args:
        domain_weight (float): Nonnegative scale for each domain loss.
            One reproduces the reference's unweighted sum. (default: 1.0)
    """

    def __init__(self, domain_weight: float = 1.0):
        super().__init__()
        if (
            isinstance(domain_weight, bool)
            or not isinstance(domain_weight, (int, float))
            or not math.isfinite(domain_weight)
            or domain_weight < 0
        ):
            raise ValueError('domain_weight must be finite and nonnegative.')
        self.domain_weight = float(domain_weight)

    def forward(self, outputs, source_targets: torch.Tensor):
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
            raise ValueError('Supply the complete source/target four-head output.')
        scores = outputs[0]
        if (
            scores.ndim != 2
            or source_targets.ndim != 1
            or len(source_targets) != len(scores)
        ):
            raise ValueError('Source targets must match source classification rows.')
        total = outputs[1].shape[0]
        source_count = scores.shape[0]
        if total <= source_count or any(
            head.ndim != 2 or head.shape != (total, 2) for head in outputs[1:]
        ):
            raise ValueError(
                'Each binary domain head needs all source and target rows.'
            )
        labels = torch.cat(
            [
                torch.zeros(source_count, dtype=torch.long, device=scores.device),
                torch.ones(
                    total - source_count, dtype=torch.long, device=scores.device
                ),
            ]
        )
        loss = nn.functional.cross_entropy(scores, source_targets)
        for head in outputs[1:]:
            loss = loss + self.domain_weight * nn.functional.cross_entropy(head, labels)
        return loss
