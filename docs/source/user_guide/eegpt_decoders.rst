EEGPT temporal decoders
=======================

:class:`torcheeg.models.EEGPT` returns summaries with shape
``[batch, temporal_patches, summary_tokens, encoder_width]``.
:class:`torcheeg.models.EEGPTReconstructor` and
:class:`torcheeg.models.EEGPTPredictor` process these summaries with the
full-token temporal transformers from the official EEGPT downstream code.

Both modules prepend a class token. All summaries in a temporal patch share
its rotary position. The default divisor is two. Rotation affects only the
first part of each attention head, as in the official implementation.
The modules compute frequencies on the current device and dtype without a
mutable cache.

Choose the representation
--------------------------

The normal output has shape
``[batch, 1 + temporal_patches * summary_tokens, hid_channels]``. It is the
hidden token sequence before the optional final normalization and projection.
With ``use_out_proj=True``, the reconstructor produces ``patch_size`` features
per token; the predictor produces ``in_features`` features per token.
These projected tokens alone are not reconstructed EEG waveforms.

To pool patch tokens, exclude the initial class token::

    import torch
    from torcheeg.models import EEGPT, EEGPTReconstructor

    encoder = EEGPT(num_electrodes=3, channel_names=['C3', 'CZ', 'C4'],
                    hid_channels=32, depth=2, num_heads=4, embed_num=2)
    decoder = EEGPTReconstructor(num_patches=encoder.num_patches,
        in_features=32, embed_num=2, hid_channels=32, depth=2, num_heads=4)
    summaries = encoder(torch.randn(2, 3, 1024))
    tokens = decoder(summaries)
    pooled = tokens[:, 1:].mean(dim=1)

This example uses random weights. The modules add no channel interpolation,
resampling, task loss or classifier. In particular, the direct summary heads
:class:`torcheeg.models.EEGPTClassifier`,
:class:`torcheeg.models.EEGPTRegression` and
:class:`torcheeg.models.EEGPTEmbedding` do not silently acquire a decoder.
Select the desired path explicitly.

Native tensor names and unused parameters
-----------------------------------------

Decoder state dictionaries retain the official reconstructor/predictor
component keys. Use ``load_state_dict`` with a complete compatible component
state. This addition does not download weights or provide an implicit loader
for arbitrary checkpoint files. Use ``torch.load(..., weights_only=True)``
for a trusted local tensor checkpoint and select its component explicitly.

The official full-token downstream forward does not use ``mask_token`` or
the reconstructor's ``chan_embed``. These parameters remain in the state for
compatibility and receive no gradients in this forward. The final projection
and normalization receive gradients only when ``use_out_proj=True``.
Neither module implements masked pretraining or its loss. Attention
inspection returns attention probabilities instead of token features.

Runnable workflows
-------------------

See :doc:`../auto_examples/examples_bciciv2b_eegpt_reconstructor` and
:doc:`../auto_examples/examples_bciciv2b_eegpt_predictor`. Each standalone script
uses MOABB preprocessing, trains a small random encoder and decoder, then
evaluates one held-out subject. These are functional teaching protocols;
they do not establish pretrained EEGPT or full EEG-Arena benchmark scores.

Source and validation
----------------------

The modules adapt the Apache-2.0 licensed
`official EEGPT downstream source <https://github.com/BINE022/EEGPT/blob/ea7282f698c406cd423ccbedbe9960853761144c/downstream/Modules/models/EEGPT_mcae_finetune.py>`_.
The retained license is ``LICENSES/EEGPT.txt``. Independent float64 fixtures
from that unmodified source check both decoders, projected and hidden
outputs, optional input/position embeddings, attention, input gradients and
all parameter gradients, including intentionally unused parameters.
