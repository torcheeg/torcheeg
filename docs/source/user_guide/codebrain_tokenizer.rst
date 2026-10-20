CodeBrain TFDual tokenizer
==========================

``CodeBrainTokenizer`` is CodeBrain's first-stage time/frequency tokenizer.
It keeps the complete encoder, two EMA codebooks, two transformer decoders
and three reconstruction heads. It can generate the target IDs consumed by
the :doc:`CodeBrain EEGSSM encoder <codebrain>` or train with the complete
six-term self-supervised objective.

Source and weights
------------------

Cite Ma et al., `CodeBrain: Bridging Decoupled Tokenizer and Multi-Scale
Architecture for EEG Foundation Model <https://arxiv.org/abs/2506.09110>`_,
ICLR 2026. The `author implementation <https://github.com/jingyingma01/CodeBrain>`_
at ``22d350caf68246d2fda4f630ef837420db3fb130`` and
`official weights <https://huggingface.co/YjMajy/CodeBrain>`_ use Apache-2.0.
Attribution and license notices are in ``LICENSES/CodeBrain.txt``.

Download ``CodeBrain_Tokenizer.pth`` from revision
``bef08d2fdb1759685371cc635aad21ce59163689``. Its SHA256 is
``e9560b670d64ea4712fd99a48dc2131326b919744c7d3eb504cf57b1ef3af999``.
The complete model state has 325 tensors, including BatchNorm statistics,
both codebook embeddings and EMA/usage buffers. The tokenizer has a 12-block,
200-wide encoder, two three-block decoders, two 4096-entry codebooks and
32-wide quantized vectors. It is distinct from ``CodeBrain.pth``; do not
interchange these checkpoints.

.. code-block:: python

   import torch
   from torcheeg.models import CodeBrainTokenizer

   tokenizer = CodeBrainTokenizer.from_pretrained(
       './CodeBrain_Tokenizer.pth', update_codebook=False, track_usage=False
   ).eval()
   eeg = torch.randn(2, 19, 4, 200)  # synthetic shape check
   with torch.no_grad():
       time_ids, frequency_ids = tokenizer.get_codebook_indices(eeg)
       # each [2,76], integer IDs in [0,4096)
       encoded = tokenizer.encode(eeg, contrastive=False)
       raw = tokenizer.decode_t(encoded[0])  # [2,76,200]
       amplitude, phase = tokenizer.decode_f(encoded[3])

Loading uses CPU and ``weights_only=True``, with no download, remote code,
unrestricted pickle fallback or partial matching. The original file also
contains optimizer and training metadata. Known Namespace, NumPy dtype and
NumPy scalar metadata are discarded through inert safe-global replacements:
the original metadata constructors and NumPy functions are not executed.
Unknown globals remain rejected. All model keys, shapes, dtypes, finite
values and initialized-codebook flags are required.

Original-file metadata handling needs ``torch.serialization.safe_globals``.
For older runtimes, make a tensor-only export once on a supporting runtime:

.. code-block:: python

   torch.save(tokenizer.state_dict(), './tokenizer-tensors.pt')
   # Later, on the target runtime:
   restored = CodeBrainTokenizer.from_pretrained(
       './tokenizer-tensors.pt', update_codebook=False, track_usage=False
   ).eval()

A complete local safetensors export is also supported with the
``foundation`` extra. Every tensor must remain present; a file with only the
encoder is not a complete tokenizer checkpoint.

Inputs and positional rows
--------------------------

Input floating tensors have shape ``[batch,electrodes,patches,200]`` at
200 Hz, with nonoverlapping one-second patches. No filtering, scaling,
normalization, channel selection or named-montage mapping is inferred.
The shared encoder combines a temporal convolution branch and a convolution
branch over the **real part of the full FFT**. This differs from the
EEGSSM encoder's magnitude-rFFT patch path.

``channel_ids`` is one distinct integer row in 1–128 per electrode, supplied
as a list or integer tensor. Row 0 is reserved for the class token. The
default assigns rows 1 through the input electrode count, matching the
author's sequential assignment. The pretraining entry point uses rows 1–19;
the presence of 128 possible rows does not prove that every row was trained
or identify a physical electrode. Keep the input order and mapping explicit.
The same row assignment must be used when encoding and decoding.

There can be at most 60 temporal patches. Unlike the source's ambiguous
decoder-axis heuristic, TorchEEG explicitly distinguishes code width,
electrodes and patches, so a single temporal patch can be reconstructed.

``encode`` returns time quantized vectors, flattened time IDs, time
quantization loss, frequency quantized vectors, flattened frequency IDs,
frequency quantization loss and contrastive loss. Quantized vectors have
shape ``[batch,code_dim,electrodes,patches]``; token IDs follow batch,
electrode and patch order. ``get_tokens`` returns named IDs and latents.
Token getters skip the unused contrastive computation by default.

Codebook updates and inference
------------------------------

The codebook embeddings do not receive optimizer gradients. In train mode,
``update_codebook=True`` updates their normalized vectors by EMA, while
ordinary neural parameters receive gradients through straight-through
quantization. New constructor-created codebooks can use native k-means
initialization from data; pretrained loading requires initialized codebooks.
When distributed training is initialized, new centroids are selected from
rank 0's batch and broadcast before token assignment. This fixes divergent
rank-local initial codebooks; subsequent counts and embedding sums are
all-reduced. A two-rank CPU Gloo regression checks synchronized initialization
and EMA updates, including module construction before process-group setup.
It is not a multi-GPU throughput measurement.

``track_usage=True`` reproduces the source's EMA usage counting even in
eval mode. It updates usage buffers but not codebook vectors during eval.
Use ``update_codebook=False, track_usage=False`` and ``eval()`` for an
immutable teacher, together with ``torch.no_grad()``. The example above uses
that configuration. ``eval()`` alone does not turn off autograd.

Complete training objective
---------------------------

``tokenizer(eeg)`` returns a scalar loss and detached log values. The six
equally weighted terms are time quantization, frequency quantization, raw
reconstruction, standardized FFT-amplitude reconstruction, standardized
FFT-phase reconstruction and temporal-half contrastive loss. FFT target
standard deviations use a 1e-6 floor to keep constant-signal losses finite.
Reconstruction heads predict the standardized amplitude and phase targets;
they do not directly return an inverse-FFT waveform.
Keep FFT precision and backend fixed when comparing phase losses. For very
small coefficients and angles near the positive/negative pi boundary, a
float64 FFT can produce a different phase target from the defined float32
Torch FFT, even when the raw input and model predictions are identical.

In ``forward`` and ``encode``, ``contrastive=None`` follows the author's
autograd-dependent behavior: the term is computed whenever gradients are
enabled, independently of train/eval mode. It requires an even patch count
of at least two. Use ``contrastive=False`` for one or odd patch counts,
including reconstruction with gradients. ``torch.no_grad()`` also disables
the automatic contrastive term.

The default ``contrastive_mode='infonce'`` fixes a source mask error: the
paired positive appears exactly once and is excluded from negative pairs.
The source subtracts the identical positive mask twice, leaving negative
values that convert back to true, so positives also appear as negatives.
Choose ``contrastive_mode='native'`` only to reproduce that objective.
Both choices retain all pretrained weights, IDs and reconstruction paths.
For a batch of one paired example, corrected InfoNCE is zero; the native
duplicate-positive term is log(2).

The corrected objective uses a fixed-size score matrix. On Torch 2.6 CPU,
the complete six-term loss, input gradients and every parameter gradient
were checked against eager execution with ``torch.compile``, fullgraph and
the ``aot_eager`` backend. That check used fixed sequential rows, initialized
codebooks, eval mode and disabled usage tracking. It does not establish
optimized-backend performance or arbitrary dynamic-metadata compilation.

Use tokens for EEGSSM pretraining
---------------------------------

The two models consume the same preprocessed patch layout. A frozen teacher
provides clean-input targets; the EEGSSM encoder receives masked patches:

.. code-block:: python

   from torch.nn import functional as F
   from torcheeg.models import CodeBrain

   encoder = CodeBrain.from_pretrained('./CodeBrain.pth')
   with torch.no_grad():
       time_ids, frequency_ids = tokenizer.get_codebook_indices(eeg)
   mask = torch.zeros(eeg.shape[:-1], dtype=torch.bool)
   mask[:, :, 0] = True
   time_logits, frequency_logits = encoder.predict_tokens(eeg, mask)
   loss = F.cross_entropy(time_logits, time_ids.reshape_as(mask)[mask])
   loss += F.cross_entropy(frequency_logits, frequency_ids.reshape_as(mask)[mask])
   loss.backward()

Keep teacher and encoder vocabularies, input scaling, electrode order and
patch order consistent. Never produce targets from already masked teacher
inputs. This snippet demonstrates one training step, not a benchmark score.

Independent real-data workflow
------------------------------

See :doc:`../auto_examples/examples_eegmat_codebrain_tokenizer` for a fixed
EEGMat workflow that fine-tunes the complete pretrained tokenizer on all
training subjects and holds out the last complete subject. Condition labels
are recorded but are not training targets. Preprocessing is whole-record
resampling to 200 Hz, FIR 0.3–75 Hz/notch 50 Hz, microvolts divided by 100,
and four one-second patches per window. This teaching protocol does not
claim to reproduce the pretraining montage, trimming or paper scores.

The export retains all model, BatchNorm and EMA state; tensor optimizer
state with JSON parameter groups; RNG and loader-generator states; input
provenance; token IDs; and all three reconstruction outputs. The script
strictly restores the complete model and optimizer and requires identical
held-out predictions. The tests also restore and reproduce the next training
step, including optimizer momentum, model buffers and sample order. Gallery
rendering does not execute this training workflow.
