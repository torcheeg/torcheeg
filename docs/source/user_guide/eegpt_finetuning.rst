EEGPT full task models
======================

:class:`torcheeg.models.EEGPTFinetuneClassifier` joins a spatial EEGPT encoder
and a temporal reconstructor or predictor. It follows the full-token task
path in the official EEGPT downstream implementation. This differs from
:class:`torcheeg.models.EEGPTClassifier`, which classifies spatial summaries
directly. Both APIs remain explicit.

Construct the task path
-----------------------

Provide an encoder with ordered channel names and a matching decoder. Their
temporal patch count, summaries per patch, encoder width, dtype and device
must agree. For example::

    import torch
    from torcheeg.models import EEGPT, EEGPTReconstructor, EEGPTFinetuneClassifier

    encoder_config = dict(num_electrodes=3, chunk_size=1024,
        channel_names=['C3', 'CZ', 'C4'], hid_channels=32,
        depth=2, num_heads=4, embed_num=2)
    decoder_config = dict(num_patches=16, in_features=32,
        hid_channels=32, depth=2, num_heads=4, embed_num=2)
    model = EEGPTFinetuneClassifier(EEGPT(**encoder_config),
        EEGPTReconstructor(**decoder_config), num_classes=2)
    logits = model(torch.randn(2, 3, 1024))

The example uses small random components. The canonical large architecture
uses width 512, eight blocks and four summaries in each component. Use
``EEGPTPredictor`` instead of ``EEGPTReconstructor`` for the predictor path.
With ``use_out_proj=True``, the reconstructor emits ``patch_size`` features
per token and the predictor emits ``in_features`` features per token.
See :doc:`eegpt_decoders` for the intentionally unused native mask tensors.
This task model implements no masked pretraining objective.

Pool and inspect features
--------------------------

The default mean pooling excludes the decoder's class token, averages the
remaining tokens, then applies LayerNorm. With ``mean_pooling=False``, it
normalizes every token and selects the class token. Final pooling LayerNorm
uses epsilon ``1e-5`` by default; component LayerNorm uses ``1e-6`` by default.
The head applies a per-class weight norm constraint before returning logits.
Set ``num_classes=0`` to return pooled features through an identity head.

``forward_features(x)`` returns pooled decoder features. Optional
``return_all_tokens=True`` and ``return_patch_tokens=True`` return normalized
tokens instead; request only one. ``decoder_feature_dim`` is their final
feature width. ``feature(x, flatten=False)`` retains the direct-head API's
raw spatial summaries before decoding; its flattened width is ``feature_dim``.

Control preprocessing and freezing
-----------------------------------

By default input shape is ``[batch, encoder_channels, encoder_samples]``.
A four-dimensional input is flattened across its last two axes in time order.
Optional ``in_channels`` adds the official constrained pointwise channel
mapping into the encoder slots. Those learned slots do not establish physical
electrode recovery. Optional ``resize_input=True`` applies nearest-neighbor
length normalization to ``encoder.chunk_size``. It does not infer sampling
rate, perform antialiasing, or guarantee a common physical duration. Set
sampling rate, time windows, reference and units in the data pipeline.
For physical resampling use an explicit preprocessing transform instead.

``freeze_encoder`` and ``freeze_decoder`` freeze parameters and keep the
selected component in evaluation mode during task training. Gradients still
flow through it into an input adapter. ``train_cls_token=True`` requires a
frozen decoder and trains only its class token within that decoder. Without
freeze flags, existing component parameter flags remain unchanged. The
channel mapper, pooling norm and classifier retain their independent flags.

Load component checkpoints safely
----------------------------------

The joint factory reads a local checkpoint once using
``torch.load(..., weights_only=True, map_location='cpu')``. Supply explicit
encoder and decoder configurations and non-overlapping prefixes::

    model = EEGPTFinetuneClassifier.from_pretrained(
        'local-large.pt',
        encoder_kwargs=dict(num_electrodes=3, chunk_size=1024,
                            channel_names=['C3', 'CZ', 'C4']),
        decoder_kwargs=dict(num_patches=16),
        decoder_type='reconstructor',
        encoder_prefix='target_encoder.',
        decoder_prefix='reconstructor.',
        num_classes=2)

This code requires an existing complete compatible checkpoint. The factory
checks every selected tensor name, shape, floating dtype and finite value.
It rejects missing and extra component tensors. Other checkpoint components
are ignored, but unsupported pickle objects anywhere in the file are rejected;
there is no unsafe fallback or automatic download. Encoder sensor order and
preprocessing still require verified configuration: tensor shapes alone
cannot establish their meaning.

The task's pooling norm, optional channel mapper and head are new parameters.
To restore a full trained task, save its complete ``state_dict`` and constructor
settings, construct the same encoder, decoder and task, then call
``load_state_dict(..., strict=True)`` on the safely loaded tensor state.
Task state keys use ``encoder.`` and ``decoder.``; the component factory uses
explicit prefixes to select native pretraining components. They are distinct
checkpoint contracts.

Run complete workflows
-----------------------

:doc:`../auto_examples/examples_bciciv2b_eegpt_finetune_reconstructor` and
:doc:`../auto_examples/examples_bciciv2b_eegpt_finetune_predictor` train and
evaluate the two task paths with BCIC-IV-2B. Each saves its complete tensor
state, constructor settings, preprocessing, subject IDs and held-out logits
and targets. The examples select small random models by default. Their
one-epoch results establish functionality, not pretrained or EEG-Arena scores.
The full EEG-Arena schedule, optimizer groups and task-specific preprocessing
must be configured separately.

Source and independent checks
------------------------------

This adapts the Apache-2.0
`official EEGPT task implementation <https://github.com/BINE022/EEGPT/blob/ea7282f698c406cd423ccbedbe9960853761144c/downstream/Modules/models/EEGPT_mcae_finetune.py>`_.
The retained license and adaptation notice are in ``LICENSES/``. Independent
float64 fixtures from the unmodified canonical 512-wide, eight-block encoder
and decoder verify full-model logits, input gradients and every parameter
gradient. They cover reconstructor mean pooling and predictor class-token
pooling with projection, learned channel mixing and nearest-neighbor resizing.
Separate smaller functional tests cover both pooling/projection choices,
freezing, optimizer ownership and complete checkpoint validation.
