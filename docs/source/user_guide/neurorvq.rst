NeuroRVQ with an external backend
=================================

``NeuroRVQ`` wraps a complete explicitly supplied author backend.
``NeuroRVQClassifier`` adds a new task head for either frozen features or
encoder fine-tuning. The adapter never downloads or imports author source.

Source and complete checkpoint
------------------------------

Cite Barmpas et al., `NeuroRVQ <https://arxiv.org/abs/2510.13068>`_.
The `author repository <https://github.com/KonstantinosBarmpas/NeuroRVQ>`_
and `official weights <https://huggingface.co/ntinosbarmpas/NeuroRVQ>`_
use **CC-BY-NC-4.0**. Obtain and use these materials separately under their
terms. TorchEEG distributes its interface, metadata persistence and new task
head; it bundles neither the author source nor the checkpoint.
See ``LICENSES/NEURORVQ-BACKEND-NOTICE.txt``.

Use author source revision ``27c3c134a53cc29c0d1696ebfe65f3e17ea9ab0c``.
The later inference-only implementation removes the prediction heads;
its permissive loader silently ignores released pretraining state. This
integration instead uses the historical complete pretraining implementation.
The EEG-v1 checkpoint at Hub revision
``d944b87f44ae0ba2923b2f10d0518f23f6803b76`` is
``pretrained_models/foundation_models/NeuroRVQ_EEG_foundation_model_v1.pt``.
Its SHA256 is
``d14ef4eae2d4594d5f1f709226a3bb5f362afd84b68050d5da0fa0586aea886a``.
It contains **330 state tensors**, 58,560,632 elements: a four-branch temporal
convolution frontend, twelve shared Transformer blocks of width 200 with ten
heads, 256 temporal positions, 105 spatial rows, and **32 independent
8192-class RVQ prediction heads**. The checkpoint size includes those heads;
the feature encoder alone is much smaller. No heads are discarded.

The example constructs the complete author model explicitly. Parameter-free
``Identity`` views expose its raw feature branches without adding random
normalization weights. All released tensors load strictly and remain in the
exported state. This is an external-backend integration, not a newly bundled
native implementation or a smaller replacement network.

Input and feature semantics
---------------------------

Input is floating ``[batch,electrodes,samples]`` at **200 Hz** with complete
200-sample patches and at most 256 patches per electrode. The caller must
provide ordered electrode names and the exact author electrode vocabulary.
Names are matched ignoring case; unknown or duplicate names are rejected.
There is no automatic channel removal, interpolation, padding or truncation.
The adapter does not infer amplitude scaling, filtering or referencing.

Temporal slots are right-aligned to the end of the 256-row table. Spatial
IDs follow the original **zero-based** vocabulary, including its use of
row zero for the first named electrode as well as the legacy class token.
The integration does not shift those IDs. Electrode/time flattening is
channel-major. The four raw contextual branches concatenate to width 800.
``forward`` averages all patch features to ``[batch,800]``;
``forward(x, return_tokens=True)`` returns ``[batch,electrodes,patches,800]``.
A new classifier produces raw logits. A frozen backbone stays in evaluation
mode while the classifier trains; ordinary fine-tuning permits gradients.

Complete tensor loading and restoration
---------------------------------------

After explicitly constructing the documented backend:

.. code-block:: python

   from torcheeg.models import NeuroRVQ, NeuroRVQClassifier

   backbone = NeuroRVQ.from_pretrained(
       './NeuroRVQ_EEG_foundation_model_v1.pt', backend,
       channel_names=channel_names, channel_vocab=original_vocabulary
   )
   model = NeuroRVQClassifier(backbone, num_classes=2, freeze_backbone=True)

The loader reads local flat tensors on CPU with ``weights_only=True``.
Every key, shape, dtype and finite value must match. Complete safetensors
and an explicit common key prefix are supported. There is no unsafe pickle
fallback, partial match or head removal. Loading weights never constructs
or executes backend code; explicit trusted backend construction is separate.

The adapter also saves spatial IDs and a fingerprint of ordered names,
original vocabulary and physical sampling rate. Strict adapter restoration
rejects a different order or vocabulary. A complete feature adapter has 332
state entries and a new two-class head adds two. Save these metadata and the
preprocessing protocol with the full state. ``to_empty`` followed by strict
restoration is covered by interface tests.

Symmetric RVQ prediction
-------------------------

``forward_masked(x, mask)`` accepts a same-device boolean mask of shape
``[batch,electrodes,patches]``. Both partitions must contain at least one
patch. It calls the original complete symmetric mask path, returning two
logit tensors shaped ``[32,masked_count,8192]`` and
``[32,unmasked_count,8192]``. Counts include all examples and electrodes;
token ordering is the native boolean-index order. All 32 heads participate
in gradients. Obtain matching target IDs from the separately supplied EEG-v1
tokenizer; the adapter does not invent targets. The complete tokenizer and its reconstruction-training example are described
below, followed by the complete 32-head masked-token workflow. The original
large-corpus training schedule is not reproduced.

Independent real EEG example and validation
-------------------------------------------

See :doc:`../auto_examples/examples_eegmat_neurorvq` for complete explicit
construction, strict weight loading, a frozen linear probe, full export and
exact prediction restoration. Obtain the author repository separately:

.. code-block:: console

   git clone https://github.com/KonstantinosBarmpas/NeuroRVQ.git ./NeuroRVQ
   git -C ./NeuroRVQ checkout 27c3c134a53cc29c0d1696ebfe65f3e17ea9ab0c
   python examples/examples_eegmat_neurorvq.py --root-path ./eegmat \
       --backend-root ./NeuroRVQ --checkpoint ./NeuroRVQ_EEG_foundation_model_v1.pt --epochs 1

The example rejects a different author revision or tracked source changes.
It preserves all 19 EEGMat EDF electrodes and source reference, resamples
whole records from 500 to 200 Hz, applies MNE FIR 0.5–45 Hz and 50-Hz notch,
and standardizes each four-second window/channel with population mean/std
and a 1e-6 floor. This is an explicit teaching transfer normalization, not a
reproduction of the original benchmark preprocessing. No population
statistics are learned from the held-out subject. The last complete subject
is held out and only a new head trains on the others for fixed epochs.
Gallery rendering does not execute the training workflow or author backend.

Independent interface tests require no author code. Optional native tests
require explicit ``TORCHEEG_NEURORVQ_BACKEND_ROOT`` and
``TORCHEEG_NEURORVQ_CHECKPOINT`` paths. They compare complete native eval/train
features, symmetric logits, EEG gradients and every parameter gradient,
and verify sample isolation. Without these paths only that external test
is skipped. A separate GitHub workflow obtains the pinned author materials
under their own terms and runs the same complete oracle.

Complete tokenizer and reconstruction training
----------------------------------------------

:class:`torcheeg.models.NeuroRVQTokenizer` accepts the **complete** original
EEG-v1 tokenizer. Use the same pinned source and Hub revisions above, with
``pretrained_models/tokenizers/NeuroRVQ_EEG_tokenizer_v1.pt``. Its SHA256 is
``d3e850e160b1a558c529466b1c0b8cebe8888a5e02824979886dd226187b2725``.
All **505 native states** load strictly: 75,988,624 stored elements including
EMA bookkeeping, with 8,355,440 trainable neural parameters. The original
12-block encoder, 3-block decoder, four eight-layer residual quantizers,
and all amplitude/sine/cosine heads remain present. Each of the 32 codebooks
has 8192 entries of width 128. The complete adapter saves 508 states, adding
electrode IDs, the metadata fingerprint, and the evaluation usage policy.

Explicit construction is shown in
:doc:`../auto_examples/examples_eegmat_neurorvq_tokenizer_training`.
The library does not import the author code or download weights:

.. code-block:: python

   from torcheeg.models import NeuroRVQTokenizer

   tokenizer = NeuroRVQTokenizer.from_pretrained(
       './NeuroRVQ_EEG_tokenizer_v1.pt', full_author_tokenizer,
       channel_names=channel_names, channel_vocab=original_vocabulary
   )
   tokenizer.train()
   result = tokenizer(eeg, return_loss=True)
   result['loss'].backward()

``get_tokens`` exposes native ``token`` IDs of shape ``[4,8,B,C*P]`` and
``indices`` of shape ``[32,B,C,P]``. Stream ``8 * branch + residual`` follows
the original four-branch, eight-residual order; the within-stream flattening
is channel-major. ``get_codebook_indices`` returns this ordered 32-stream
tensor. ``quantized`` contains all four ``[B,128,C,P]`` feature maps.

``forward_pretraining`` (or ``forward(return_loss=True)``) exposes the original
code, amplitude, phase and standardized-signal losses and their **sum**. It
calls the native encoder, RVQs, decoder and loss methods. This includes the
original residual auxiliary loss and its reduction; no alternate codebook
objective is substituted. ``original`` and ``reconstruction`` are ``[B,C,T]``,
standardized per example across all channels and times using native unbiased
variance. They are **not physical-voltage outputs**. Ordinary ``forward``
returns these two tensors. Amplitude and sine/cosine predictions are also
available in the training dictionary.

Training updates the native EMA codebooks and usage counters. Author evaluation
also changes 32 usage buffers. By default, the adapter restores those counters
after evaluation and requires already initialized codebooks. Set
``track_usage=True`` explicitly to retain the native evaluation statistics;
strict checkpoint restoration rejects a policy change. Freezing a teacher
requires both ``eval()`` and ``requires_grad_(False)``. Calls on a shared
instance must be serialized; concurrent CUDA streams have not been validated.

The adapter refreshes native EMA reduction callbacks at call time instead
of capturing process-group availability at construction. Distributed training
requires initialized codebooks. Read-only evaluation performs no statistic
collectives, permitting evaluation on one rank. Native usage-tracking evaluation
requires all ranks to enter its collectives. GPU and distributed full-model
training are not established by the CPU example.

The independent example trains all used tokenizer parameters on real EEGMat
windows with a subject holdout and native EMA updates. It checks raw-file hashes
and cache preprocessing manifests, saves the full adapter/AdamW/CPU RNG/cursor,
and supports exact continuation with a new output path. It uses all19 measured
electrodes at200Hz with the same explicit teaching normalization as the probe.
Four held-out windows are evaluated without state changes or model selection.
This is a few-step workflow check, not a published benchmark or reproduction
of the original corpus and training schedule.

Optional full-tokenizer oracle tests require
``TORCHEEG_NEURORVQ_BACKEND_ROOT`` and
``TORCHEEG_NEURORVQ_TOKENIZER_CHECKPOINT``. They compare original token order,
all four losses, standardized outputs, EEG and every used parameter gradient,
all EMA states, read-only evaluation, and exact optimizer/RNG continuation.
The dedicated GitHub workflow downloads both complete official checkpoints,
verifies their hashes, preserves their licenses, and runs the native tests.

Train all 32 masked-token prediction heads
-------------------------------------------

:class:`torcheeg.models.NeuroRVQMaskedTokenPretrainer` connects the complete
EEG-v1 backbone to the matching complete tokenizer. Supply both loaded adapters
with identical ordered electrode names, vocabulary, sampling rate and patch
geometry. The tokenizer must use ``track_usage=False``; shared parameters are
rejected. The wrapper freezes the tokenizer, keeps it in evaluation mode when
the backbone trains, and preserves every teacher state including usage buffers.
It retains both original networks: **330 backbone plus 505 tokenizer states**.
Together with physical metadata, usage policy and the objective fingerprint,
the exported wrapper has **841 state entries**. The teacher decoder remains
present even though target generation only needs encoding.

.. code-block:: python

   from torcheeg.models import NeuroRVQMaskedTokenPretrainer

   # Explicit complete adapters loaded from both matching EEG-v1 checkpoints.
   model = NeuroRVQMaskedTokenPretrainer(backbone, tokenizer)
   model.train()
   # mask: boolean [batch,electrodes,patches], same device as EEG.
   result = model(eeg, mask)
   result['loss'].backward()

The mask and its complement must both contain at least one patch across the
batch. All 32 original heads predict the corresponding frozen tokenizer IDs.
For each branch, heads ``8 * branch`` through ``8 * branch + 7`` follow the
eight native residual layers. Boolean indexing preserves batch, electrode,
then temporal order. Returned targets are ``[32,B,C,P]``; logits are
``[32,masked_count,8192]`` and ``[32,complement_count,8192]``.

The objective is explicit: for **each partition separately**, cross entropy
is averaged over its selected patches and all 32 heads. The two partition
means are then **summed**. Unequal partition sizes do not weight one view
more heavily. ``masked_loss`` and ``symmetric_loss`` expose these terms.
The fingerprint in the state dictionary guards this reduction policy.

This reduction is a TorchEEG training interface using the complete released
networks, targets and logits. The pinned author repository exposes symmetric
prediction but does not release its stage-two trainer or specify this loss
reduction. The paper describes symmetric masking. This interface does not
claim to reproduce an unavailable training loop or the original full-corpus
schedule. No targets, codebooks or prediction heads are randomly substituted.

Independent real-data workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`../auto_examples/examples_eegmat_neurorvq_mtp_training`.
The standalone script requires the explicit complete initial checkpoint pair,
or a full strict resume artifact. It uses the same documented EEGMat teaching
preprocessing as the tokenizer workflow and holds out the final sorted subject.
Each example masks ``floor(C*P/2)`` patches without replacement using a dedicated
CPU generator; the complement forms the second view. This mask-generation
policy is explicit and is not attributed to an unpublished author sampler.

.. code-block:: console

   python examples/examples_eegmat_neurorvq_mtp_training.py --root-path ./eegmat \
       --backend-root ./NeuroRVQ \
       --foundation-checkpoint ./NeuroRVQ_EEG_foundation_model_v1.pt \
       --tokenizer-checkpoint ./NeuroRVQ_EEG_tokenizer_v1.pt \
       --steps 2 --output-path ./mtp2.pt
   python examples/examples_eegmat_neurorvq_mtp_training.py --root-path ./eegmat \
       --backend-root ./NeuroRVQ --resume ./mtp2.pt --steps 3 --output-path ./mtp3.pt

``--steps`` is the total including resumed steps. The example saves all 841
model states, backbone AdamW state, global CPU and mask-generator RNG, next
batch cursor, input hashes and preprocessing protocol. Resume rejects protocol
changes. It checks cache manifests and never overwrites an existing output.
Validation uses the first four held-out windows with a separate fixed mask
generator, without parameter selection or teacher updates. The few-step CPU
run checks the workflow; its losses are not benchmark results.

Normal interface tests use clearly labeled protocol doubles. Optional complete
native tests require both checkpoint environment variables above. They compare
all 32-stream targets, both native logit tensors, explicit mathematical CE,
EEG and all 330 backbone parameter gradients, immutable teacher state, and
exact full optimizer/global-RNG/mask-RNG continuation. These are complete
original models, not smaller numerical substitutes. GPU execution, concurrent
shared-model calls and a full distributed training benchmark remain unverified.
