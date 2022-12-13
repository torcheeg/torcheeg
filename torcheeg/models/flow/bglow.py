import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_p(mean, logs, x):
    c = math.log(2 * math.pi)
    return -0.5 * (logs * 2.0 + ((x - mean)**2) / torch.exp(logs * 2.0) + c)


def gaussian_likelihood(mean, logs, x):
    p = gaussian_p(mean, logs, x)
    return torch.sum(p, dim=[1, 2, 3])


def gaussian_sample(mean, logs, temperature=1):
    z = torch.normal(mean, torch.exp(logs) * temperature)

    return z


def squeeze2d(input, factor):
    if factor == 1:
        return input

    B, C, H, W = input.size()

    assert H % factor == 0 and W % factor == 0, "H or W modulo factor is not 0"

    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)

    return x


def unsqueeze2d(input, factor):
    if factor == 1:
        return input

    factor2 = factor**2

    B, C, H, W = input.size()

    assert C % (factor2) == 0, "C module factor squared is not 0"

    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)

    return x


class _ActNorm(nn.Module):
    def __init__(self, num_features, scale=1.0):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.scale = scale
        self.inited = False

    def initialize_parameters(self, input):
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not inited")

        with torch.no_grad():
            bias = -torch.mean(input.clone(), dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((input.clone() + bias)**2,
                              dim=[0, 2, 3],
                              keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)

            self.inited = True

    def _center(self, input, reverse=False):
        if reverse:
            return input - self.bias
        else:
            return input + self.bias

    def _scale(self, input, logdet=None, reverse=False):

        if reverse:
            input = input * torch.exp(-self.logs)
        else:
            input = input * torch.exp(self.logs)

        if logdet is not None:
            b, c, h, w = input.shape

            dlogdet = torch.sum(self.logs) * h * w

            if reverse:
                dlogdet *= -1

            logdet = logdet + dlogdet

        return input, logdet

    def forward(self, input, logdet=None, reverse=False):
        self._check_input_dim(input)

        if not self.inited:
            self.initialize_parameters(input)

        if reverse:
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)
        else:
            input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)

        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.0):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()))


class LinearZeros(nn.Module):
    def __init__(self, in_channels, out_channels, logscale_factor=3):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.logscale_factor = logscale_factor

        self.logs = nn.Parameter(torch.zeros(out_channels))

    def forward(self, input):
        output = self.linear(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Conv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
            do_actnorm=True,
            weight_std=0.05,
    ):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=(not do_actnorm),
        )

        # init weight with std
        self.conv.weight.data.normal_(mean=0.0, std=weight_std)

        if not do_actnorm:
            self.conv.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)

        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = self.conv(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
            logscale_factor=3,
    ):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, input):
        output = self.conv(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle):
        super().__init__()
        self.num_channels = num_channels
        self.indices = torch.arange(self.num_channels - 1,
                                    -1,
                                    -1,
                                    dtype=torch.long)
        self.indices_inverse = torch.zeros((self.num_channels),
                                           dtype=torch.long)

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        shuffle_idx = torch.randperm(self.indices.shape[0])
        self.indices = self.indices[shuffle_idx]

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, input, reverse=False):
        assert len(input.size()) == 4

        if not reverse:
            input = input[:, self.indices, :, :]
            return input
        else:
            return input[:, self.indices_inverse, :, :]


class Split2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = Conv2dZeros(num_channels // 2, num_channels)

    def split2d_prior(self, z):
        h = self.conv(z)
        return split_feature(h, "cross")

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            z1 = input
            mean, logs = self.split2d_prior(z1)
            z2 = gaussian_sample(mean, logs, temperature)
            z = torch.cat((z1, z2), dim=1)
            return z, logdet
        else:
            z1, z2 = split_feature(input, "split")
            mean, logs = self.split2d_prior(z1)
            logdet = gaussian_likelihood(mean, logs, z2) + logdet
            return z1, logdet


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if reverse:
            output = unsqueeze2d(input, self.factor)
        else:
            output = squeeze2d(input, self.factor)

        return output, logdet


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(self.lower.device)
            self.eye = self.eye.to(self.lower.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous().to(
                self.upper.device)
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1,
                           1).to(input.device), dlogdet.to(input.device)

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)

        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(
        kernel_size
    ), "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def uniform_binning_correction(x, n_bits=8):
    b, c, h, w = x.size()
    n_bins = 2**n_bits
    chw = c * h * w
    x += torch.zeros_like(x).uniform_(0, 1.0 / n_bins)

    objective = -math.log(n_bins) * chw * torch.ones(b, device=x.device)
    return x, objective


def split_feature(tensor, type="split"):
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def get_block(in_channels, out_channels, hid_channels):
    block = nn.Sequential(
        Conv2d(in_channels, hid_channels),
        nn.ReLU(inplace=False),
        Conv2d(hid_channels, hid_channels, kernel_size=(1, 1)),
        nn.ReLU(inplace=False),
        Conv2dZeros(hid_channels, out_channels),
    )
    return block


class FlowStep(nn.Module):
    def __init__(
        self,
        in_channels,
        hid_channels,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
    ):
        super().__init__()
        self.flow_coupling = flow_coupling

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels,
                                             LU_decomposed=LU_decomposed)
            self.flow_permutation = lambda z, logdet, rev: self.invconv(
                z, logdet, reverse=rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
            self.flow_permutation = lambda z, logdet, rev: (
                self.shuffle(z, reverse=rev),
                logdet,
            )
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
            self.flow_permutation = lambda z, logdet, rev: (
                self.reverse(z, reverse=rev),
                logdet,
            )
        if flow_coupling == "additive":
            self.block = get_block(in_channels // 2, in_channels // 2,
                                   hid_channels)
        elif flow_coupling == "affine":
            self.block = get_block(in_channels // 2, in_channels, hid_channels)

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)

        z, logdet = self.flow_permutation(z, logdet, False)

        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        z, logdet = self.flow_permutation(z, logdet, True)

        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        return z, logdet


class FlowNet(nn.Module):
    def __init__(
        self,
        in_channels,
        grid_size,
        hid_channels,
        num_layers,
        num_blocks,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.output_shapes = []

        self.num_layers = num_layers
        self.num_blocks = num_blocks

        H, W = grid_size
        C = in_channels

        for i in range(num_blocks):
            C, H, W = C * 4, H // 2, W // 2
            self.layers.append(SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])

            for _ in range(num_layers):
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hid_channels=hid_channels,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                    ))
                self.output_shapes.append([-1, C, H, W])

            if i < num_blocks - 1:
                self.layers.append(Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        else:
            return self.encode(input, logdet)

    def encode(self, z, logdet=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z,
                                  logdet=0,
                                  reverse=True,
                                  temperature=temperature)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
        return z


class BGlow(nn.Module):
    r'''
    A flow-based model is dedicated to train an encoder that encodes the input as a hidden variable and makes the hidden variable obey the standard normal distribution. By good design, the encoder should be reversible. On this basis, as soon as the encoder is trained, the corresponding decoder can be used to generate samples from a Gaussian distribution according to the inverse operation. In particular, the Glow model is a easy-to-use flow-based model that replaces the operation of permutating the channel axes by introducing a 1x1 reversible convolution.

    - Paper: Kingma D P, Dhariwal P. Glow: Generative flow with invertible 1x1 convolutions[J]. Advances in neural information processing systems, 2018, 31.
    - URL: https://arxiv.org/abs/1807.03039
    - Related Project: https://github.com/y0ast/Glow-PyTorch

    Below is a recommended suite for use in EEG generation:

    .. code-block:: python

        eeg = torch.randn(1, 4, 32, 32)
        model = BGlow()
        z, nll_loss, y_logits = model(eeg)
        fake_X = model(temperature=1.0, reverse=True)

    Below is a recommended suite for use in conditional EEG generation:

    .. code-block:: python

        eeg = torch.randn(1, 4, 32, 32)
        y = torch.randint(0, 2, (2, ))
        model = BGlow(num_classes=2)
        z, nll_loss, y_logits = model(eeg, y)
        fake_X = model(y=y, temperature=1.0, reverse=True)

    Args:
        in_channels (int): The feature dimension of each electrode. (defualt: :obj:`4`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (defualt: :obj:`(9, 9)`)
        hid_channels (int): The basic hidden channels in the network blocks. (defualt: :obj:`512`)
        num_layers (int): The number of steps in the flow, each step contains an affine coupling layer, an invertible 1x1 conv and an actnorm layer. (defualt: :obj:`32`)
        num_blocks (int): Number of blocks, each block includes split, step of flow and squeeze. (defualt: :obj:`3`)
        actnorm_scale (float): The pre-defined scale factor in the actnorm layer. (defualt: :obj:`1.0`)
        flow_permutation (str): The used flow permutation method, options include :obj:`invconv`, :obj:`shuffle` and :obj:`reverse`. (defualt: :obj:`invconv`)
        flow_coupling (str): The used flow coupling method, options include :obj:`additive` and  :obj:`affine`. (defualt: :obj:`affine`)
        LU_decomposed (bool): Whether to use LU decomposed 1x1 convs. (defualt: :obj:`True`)
        learnable_prior (bool): Whether to train top layer (prior). (defualt: :obj:`True`)
        num_classes (int): The number of classes. If the number of categories is greater than 0, conditional Glow will be used. During the generation process, additional category labels are provided to guide the generation of samples for the specified category. (defualt: :obj:`-1`)
    '''
    def __init__(self,
                 in_channels: int = 4,
                 grid_size: Tuple[int, int] = (32, 32),
                 hid_channels: int = 512,
                 num_layers: int = 32,
                 num_blocks: int = 3,
                 actnorm_scale: float = 1.0,
                 flow_permutation: str = "invconv",
                 flow_coupling: str = "affine",
                 LU_decomposed: bool = True,
                 learnable_prior: bool = True,
                 num_classes: int = -1):
        super(BGlow, self).__init__()
        self.num_classes = num_classes
        self.flow = FlowNet(
            in_channels=in_channels,
            grid_size=grid_size,
            hid_channels=hid_channels,
            num_layers=num_layers,
            num_blocks=num_blocks,
            actnorm_scale=actnorm_scale,
            flow_permutation=flow_permutation,
            flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed,
        )
        self.num_classes = num_classes
        self.condition = num_classes > 0

        self.learnable_prior = learnable_prior

        # learned prior
        if learnable_prior:
            C = self.flow.output_shapes[-1][1]
            self.learnable_prior_fn = Conv2dZeros(C * 2, C * 2)

        if self.condition:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = LinearZeros(num_classes, 2 * C)
            self.project_class = LinearZeros(C, num_classes)

        self.register_buffer(
            "prior_h",
            torch.zeros([
                1,
                self.flow.output_shapes[-1][1] * 2,
                self.flow.output_shapes[-1][2],
                self.flow.output_shapes[-1][3],
            ]),
        )

    def prior(self, data, y_onehot=None):
        if data is not None:
            h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        else:
            if not y_onehot is None:
                batch_size = y_onehot.shape[0]
            else:
                batch_size = 32
            # Hardcoded a batch size of 32 here
            h = self.prior_h.repeat(batch_size, 1, 1, 1)
        channels = h.size(1)

        if self.learnable_prior:
            h = self.learnable_prior_fn(h)

        if self.condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot)
            h += yp.view(h.shape[0], channels, 1, 1)

        return split_feature(h, "split")

    def forward(self,
                x: torch.Tensor = None,
                labels: torch.Tensor = None,
                z: torch.Tensor = None,
                temperature: float = 1.0,
                reverse: bool = False):
        r'''
        Args:
            x (torch.Tensor): EEG signal representation. The ideal input shape is :obj:`[n, 4, 9, 9]`. Here, :obj:`n` corresponds to the batch size, :obj:`4` corresponds to the :obj:`in_channels`, and :obj:`(9, 9)` corresponds to the :obj:`grid_size`.
            labels (torch.Tensor): Category labels (int) for a batch of samples The shape should be :obj:`[n,]`. Here, :obj:`n` corresponds to the batch size.
            z (torch.Tensor): The latent vector :obj:`z`. If it is not passed in, then it is automatically sampled from a gaussian distribution.
            temperature (float): The hyper-parameter, temperature, to sample from gaussian distributions. (defualt: :obj:`1.0`)
            reverse (bool): forward process (False) or reverse process (True). Among them, the forward process is used to extract the hidden variables from the EEG representation for training. The reverse process is used to sample the hidden variables from the random distribution for inverse operation to get the generated samples. With :obj:`reverse=True`, :obj:`y`, :obj:`z`, :obj:`temperature` can be specified. In :obj:`reverse=False`, :obj:`x` must be specified.

        Returns:
            torch.Tensor: (:obj:`reverse=True`) the generated results, which should have the same shape as the input noise, i.e., :obj:`[n, 4, 9, 9]`. Here, :obj:`n` corresponds to the batch size, :obj:`4` corresponds to :obj:`in_channels`, and :obj:`(9, 9)` corresponds to :obj:`grid_size`.
            tuple: (:obj:`reverse=False`) The first term indicates the hidden variables extracted from the input. The second term is the value of the loss function for training Glow. For the third term, if a :obj:`num_classes` greater than 0 is specified, an additional classifier is used to provide the classification results of the sample. It can be used for being classified loss optimization.
        '''
        if labels is None:
            y_onehot = None
        else:
            y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()

        if reverse:
            return self.reverse_flow(z, y_onehot, temperature)
        else:
            return self.normal_flow(x, y_onehot)

    def normal_flow(self, x, y_onehot):
        b, c, h, w = x.shape

        x, logdet = uniform_binning_correction(x)

        z, objective = self.flow(x, logdet=logdet, reverse=False)

        mean, logs = self.prior(x, y_onehot)
        objective += gaussian_likelihood(mean, logs, z)

        if self.condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        bpd = (-objective) / (math.log(2.0) * c * h * w)

        return z, bpd, y_logits

    def reverse_flow(self, z, y_onehot, temperature):
        with torch.no_grad():
            if z is None:
                mean, logs = self.prior(z, y_onehot)
                z = gaussian_sample(mean, logs, temperature)
            x = self.flow(z, temperature=temperature, reverse=True)
        return x

    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True