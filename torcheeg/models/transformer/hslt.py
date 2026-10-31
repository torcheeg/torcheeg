# Adapted from XJTU-EEG/LibEER at dddff9776dbdae21195fe320dff0a5ba61628a18.
# Copyright (c) 2024 xjtu-eeg. MIT license; see LICENSES/HSLT-LIBEER-LICENSE.txt.
# Paper: Wang et al., IEEE Sensors Journal, doi:10.1109/JSEN.2022.3144317.

import torch
from torch import nn
from torch.nn import functional as F


class HSLT(nn.Module):
    r"""Complete LibEER hierarchical spatial learning Transformer.

    Nine regional electrode Transformers feed the region Transformer and
    complete classifier. Input is floating ``[batch,electrodes,features]``
    with electrodes concatenated in the documented regional order. No
    automatic reordering or feature extraction is performed. This is the
    MIT LibEER reimplementation, not a claim of original-author equivalence.

    Args:
        num_electrodes (int): Grouped DEAP 32 or SEED 62 rows. (default: 32)
        in_channels (int): Features per electrode. (default: 5)
        num_classes (int): Output classes. (default: 2)
        electrode_embed_dim (int): Electrode embedding width. (default: 8)
        region_embed_dim (int): Region embedding width. (default: 16)
        electrode_depth (int): Encoder blocks per region. (default: 2)
        region_depth (int): Region encoder blocks. (default: 2)
        return_logits (bool): Return pre-activation scores for cross entropy.
            False preserves native sigmoid for two classes and softmax for
            other class counts. (default: False)

    .. code-block:: python

        model = HSLT(return_logits=True)
        logits = model(torch.randn(2, 32, 5))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
        loss.backward()
    """

    def __init__(
        self,
        num_electrodes: int = 32,
        in_channels: int = 5,
        num_classes: int = 2,
        electrode_embed_dim: int = 8,
        region_embed_dim: int = 16,
        electrode_depth: int = 2,
        region_depth: int = 2,
        return_logits: bool = False,
    ):
        super(HSLT, self).__init__()
        for name, value in [
            ('num_electrodes', num_electrodes),
            ('in_channels', in_channels),
            ('num_classes', num_classes),
            ('electrode_embed_dim', electrode_embed_dim),
            ('region_embed_dim', region_embed_dim),
            ('electrode_depth', electrode_depth),
            ('region_depth', region_depth),
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if num_electrodes not in (32, 62):
            raise ValueError('The complete model supports 32 or 62 grouped electrodes.')
        if not isinstance(return_logits, bool):
            raise TypeError('return_logits must be a boolean.')
        self.in_channels = in_channels
        self.return_logits = return_logits
        self.classes = num_classes
        self.brain_regions = []
        self.electrode_embed_dim = electrode_embed_dim
        self.region_embed_dim = region_embed_dim
        self.electrode_depth = electrode_depth
        self.region_depth = region_depth
        self.num_electrodes = num_electrodes
        if num_electrodes == 32:
            self.brain_regions = [
                {'name': 'Pre-Frontal', 'N': 4},
                {'name': 'Frontal', 'N': 5},
                {'name': 'Left Temporal', 'N': 3},
                {'name': 'Central', 'N': 5},
                {'name': 'Right Temporal', 'N': 3},
                {'name': 'Left Parietal', 'N': 3},
                {'name': 'Parietal', 'N': 3},
                {'name': 'Right Parietal', 'N': 3},
                {'name': 'Occipital', 'N': 3},
            ]
            self.regions_electrodes_num = 3
        elif num_electrodes == 62:
            self.brain_regions = [
                {'name': 'Pre-Frontal', 'N': 5},
                {'name': 'Frontal', 'N': 9},
                {'name': 'Left Temporal', 'N': 6},
                {'name': 'Central', 'N': 7},
                {'name': 'Right Temporal', 'N': 6},
                {'name': 'Left Parietal', 'N': 8},
                {'name': 'Parietal', 'N': 4},
                {'name': 'Right Parietal', 'N': 7},
                {'name': 'Occipital', 'N': 10},
            ]
            self.regions_electrodes_num = 4
        self.regions_num = 9
        self.transformers = nn.ModuleList()
        for region in self.brain_regions:
            N = region['N']
            patch_embeddings = _LinearEmbedding(
                N, in_channels, self.electrode_embed_dim
            )
            patch_transformer = _TransformerEncoder(
                self.electrode_embed_dim, self.electrode_depth
            )
            if N != self.regions_electrodes_num:
                projection = nn.Linear(N + 1, self.regions_electrodes_num + 1)
            else:
                projection = None
            self.transformers.extend([patch_embeddings, patch_transformer, projection])
        self.regions_embeddings = _LinearEmbedding(
            self.regions_num,
            (self.regions_electrodes_num + 1) * self.electrode_embed_dim,
            self.region_embed_dim,
            False,
        )
        self.regions_transformer = _TransformerEncoder(
            self.region_embed_dim, self.region_depth
        )
        self.prediction = nn.Linear(self.region_embed_dim, self.classes)
        if self.classes == 2:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, inputs: torch.Tensor):
        if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
            raise TypeError('inputs must be a floating EEG feature tensor.')
        if inputs.ndim != 3 or inputs.shape[1:] != (
            self.num_electrodes,
            self.in_channels,
        ):
            raise ValueError(
                'Expected [batch,num_electrodes,in_channels] grouped features.'
            )
        transformer_outputs = []
        curr = 0
        for i in range(0, self.regions_num):
            patch_embeddings, patch_transformer, projection = self.transformers[
                i * 3 : (i + 1) * 3
            ]
            N = self.brain_regions[i]['N']
            x = inputs[:, curr : curr + N, :]
            transformer_output = patch_embeddings(x)
            transformer_output = patch_transformer(transformer_output)
            transformer_output = transformer_output.unsqueeze(1)
            curr = N + curr
            if N != self.regions_electrodes_num:
                transformer_output = transformer_output.permute(0, 1, 3, 2)
                transformer_output = projection(transformer_output)
                transformer_output = transformer_output.permute(0, 1, 3, 2)
            transformer_outputs.append(transformer_output)
        x = torch.cat(transformer_outputs, dim=1)
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = self.regions_embeddings(x)
        regions_outputs = self.regions_transformer(x)
        class_token_output = self.prediction(regions_outputs[:, 0, :])
        if not self.return_logits:
            class_token_output = self.activation(class_token_output)
        return class_token_output


class _LinearEmbedding(nn.Module):
    def __init__(
        self, num_patches, in_channels, projection_dim, expand=True, regions_dropout=0.1
    ):
        super(_LinearEmbedding, self).__init__()
        self.num_patches = num_patches
        self.in_channels = in_channels
        self.projection_dim = projection_dim
        self.expand = expand
        self.class_token = nn.Parameter(
            torch.randn(1, projection_dim), requires_grad=True
        )
        self.projection = nn.Linear(in_channels, projection_dim)
        self.position_embedding = nn.Embedding(num_patches + 1, projection_dim)
        self.dropout = nn.Dropout(regions_dropout)

    def forward(self, patch):
        if self.expand:
            batch = patch.size(0)
            class_token = self.class_token.expand(batch, -1)
            class_token = class_token.view(batch, 1, self.projection_dim)
            patches_embed = self.projection(patch)
            patches_embed = torch.cat([class_token, patches_embed], 1)
            positions = torch.arange(0, self.num_patches + 1, 1, device=patch.device)
            positions_embed = self.position_embedding(positions)
            encoded = patches_embed + positions_embed
        else:
            batch = patch.size(0)
            class_token = self.class_token.expand(batch, -1)
            class_token = class_token.view(batch, 1, self.projection_dim)
            patches_embed = self.projection(patch)
            patches_embed = self.dropout(patches_embed)
            patches_embed = torch.cat([class_token, patches_embed], 1)
            positions = torch.arange(0, self.num_patches + 1, 1, device=patch.device)
            positions_embed = self.position_embedding(positions)
            encoded = patches_embed + positions_embed
        return encoded


class _MLP(nn.Module):
    def __init__(self, hidden_states, output_states, dropout=0.4):
        super(_MLP, self).__init__()
        self.fc1 = nn.Linear(output_states, hidden_states)
        self.fc2 = nn.Linear(hidden_states, output_states)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = F.gelu(self.fc1(x))
        dr_hidden = self.dropout(hidden)
        output = F.gelu(self.fc2(dr_hidden))
        dr_output = self.dropout(output)
        return dr_output


class _TransformerEncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads=16, msa_dimensions=64, dropout_rate=0.4):
        super(_TransformerEncoderBlock, self).__init__()
        self.model_dim = model_dim
        self.layernormalization1 = nn.LayerNorm(model_dim, eps=1e-06)
        self.fc1 = nn.Linear(model_dim, msa_dimensions)
        self.attention = nn.MultiheadAttention(
            embed_dim=msa_dimensions,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.fc2 = nn.Linear(msa_dimensions, model_dim)
        self.layernormalization2 = nn.LayerNorm(model_dim, eps=1e-06)
        self.mlp = _MLP(model_dim * 4, model_dim)

    def forward(self, x):
        x1 = self.layernormalization1(x)
        x1 = F.relu(self.fc1(x1))
        attention_output, _ = self.attention(x1, x1, x1)
        attention_output = F.relu(self.fc2(attention_output))
        x2 = x + attention_output
        x3 = self.layernormalization2(x2)
        x3 = self.mlp(x3)
        y = x2 + x3
        return y


class _TransformerEncoder(nn.Module):
    def __init__(self, model_dim, num_blocks):
        super(_TransformerEncoder, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(_TransformerEncoderBlock(model_dim))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
