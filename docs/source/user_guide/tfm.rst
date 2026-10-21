TFM tokenizer and encoder
=========================

``TFMTokenizer`` learns single-channel time-frequency motifs and reconstructs
spectrograms. ``TFMEncoder`` uses their discrete IDs for contextual features
and predictions. ``TFMClassifier`` keeps both complete models and adds a new
task head for raw, preprocessed multi-channel EEG.

Source and complete released states
-----------------------------------

Cite Pradeepkumar et al., `Tokenizing Single-Channel EEG with Time-Frequency
Motif Learning <https://openreview.net/forum?id=2sPmWHZ8Ir>`_, ICLR 2026.
The `author source <https://github.com/Jathurshan0330/TFM-Tokenizer>`_ at
``2d6da482b16dabbb2ebec808fa9f505fc6f367c4`` and the bundled checkpoints use
MIT. See ``LICENSES/TFMTokenizer.txt`` and the shared linear-attention notices
in ``LICENSES/BIOT.txt``. The linear-attention implementation matches
``linear_attention_transformer==0.19.1`` without adding that runtime package.

The original paired **single-dataset CHB-MIT** release is in
``pretrained_weigths/single_dataset_settings/CHBMIT_tfm_tokenizer_2x2x8/``.
The complete **multi-dataset MTP-pretrained** pair is also supported below.
Other dataset-specific or seed-specific releases still require separate
architecture and checkpoint validation.

* ``tfm_tokenizer_last.pth``: all 189 tensors, SHA256
  ``093975d3eb391e5009036248e0aa94a682d47afb3a2108446be0901951075cf8``.
* ``tfm_encoder_best_model.pth``: all 59 tensors, SHA256
  ``21bf6ba5bda9b7acf78e3748af39064721c8e5753d4e36094e3ed6bdc92e48d4``.

The tokenizer has width 64, vocabulary 8192, two frequency-encoder blocks,
two temporal-encoder blocks and eight decoder blocks. Its released
reconstruction head is **one Linear(64,100)**. Current source instead enables
a two-layer MLP and retains the old linear form as a comment. TorchEEG
configures the released linear head explicitly and loads every tensor.
It does not ignore the decoder to accommodate this difference. Constructors
also accept ``reconstruction_head='mlp'`` for the separate source configuration;
a matching complete state must be supplied explicitly.

.. code-block:: python

   from torcheeg.models import TFMTokenizer, TFMEncoder

   tokenizer = TFMTokenizer.from_pretrained(
       './tfm_tokenizer_last.pth', update_codebook=False
   ).eval()
   encoder = TFMEncoder.from_pretrained('./tfm_encoder_best_model.pth').eval()

Both loaders use local CPU ``weights_only=True`` with strict keys, shapes,
dtypes and finite values. All EMA buffers, deterministic index rows,
reconstruction weights and original task-head weights are required.
Unknown extras and partial states fail. There is no download, remote code
or unrestricted pickle fallback. Complete safetensors files are supported
with the ``foundation`` extra. An explicit common ``prefix`` is supported.

Single-channel tokenization and reconstruction
----------------------------------------------

Provide a floating waveform ``[batch,samples]`` at 200 Hz and its matching
magnitude STFT ``[batch,100,frames]``. The reference STFT uses a periodic
Hann window, n_fft=200, hop_length=100, no centering, and bins 0–99, excluding
Nyquist. Thus ``frames=(samples-200)//100+1``. Keep filtering, reference,
signal units and normalization explicit; the tokenizer does not infer them.

.. code-block:: python

   import torch

   eeg = torch.randn(2, 800)  # synthetic API check
   spectrum = torch.stft(
       eeg, n_fft=200, hop_length=100, window=torch.hann_window(200),
       center=False, return_complex=True
   ).abs()[:, :100]
   with torch.no_grad():
       codes, ids, vectors = tokenizer.tokenize(spectrum, eeg)
       reconstructed, ids, straight_through, vectors = tokenizer(spectrum, eeg)
       # codes/vectors [2,7,64]; ids [2,7]; reconstructed [2,100,7]

The frequency branch uses convolution, noncausal linear attention and gated
aggregation. The temporal branch uses 200-sample convolution with stride 100.
The combined temporal encoder produces normalized vectors before nearest
codebook lookup. Reconstruction predicts the magnitude spectrum; it is not
a phase estimate or an inverse-STFT waveform.

Codebooks and training loss
---------------------------

Codebook vectors update through EMA, not optimizer gradients.
``update_codebook=True`` updates the embeddings, cluster counts and embedding
sums in train mode, **including under torch.no_grad()**. Use ``eval()`` for
an immutable teacher. When distributed training is initialized, counts and
sums are all-reduced; initialize ranks with the same checkpoint or use
PyTorch DDP to synchronize initial state. A two-process CPU Gloo test checks
that each rank matches a concatenated global batch, including module
construction before process-group setup.

``forward`` returns straight-through code vectors, so reconstruction gradients
reach the encoders. Add the EMA commitment term explicitly:

.. code-block:: python

   tokenizer.train()
   reconstructed, ids, quantized, vectors = tokenizer(spectrum, eeg)
   loss = torch.nn.functional.mse_loss(reconstructed, spectrum)
   loss += tokenizer.quantization_loss(vectors, quantized)
   loss.backward()

``quantization_loss`` is beta times MSE against detached code vectors.
The author's helper adds opposing codebook and commitment terms to the
straight-through vectors; at beta=1 their encoder gradients cancel.
TorchEEG's helper retains the commitment gradient while EMA handles the
codebook update. An analytic regression verifies that distinction. This
snippet illustrates an unmasked reconstruction step. It does not reproduce
the author's complete symmetric spectral-mask training schedule or scores.

Token encoder, channel rows and task heads
------------------------------------------

``TFMEncoder`` takes integer IDs ``[batch,electrodes,frames]``. Up to 16
channel rows and 1000 temporal positions per channel are available. Rows
0–15 are sequential source assignments, not a named physical montage.
Keep electrode order and mapping explicit. Optional ``channel_ids`` must be
distinct rows from that range. Temporal positions restart for each channel.
The final token-embedding row, ID 8192 in the release, is reserved for masking.

``forward_features`` returns the contextual class token ``[batch,64]``.
The single-CHB-MIT encoder has a **single-logit binary classification head**;
``forward`` returns ``[batch,1]`` without sigmoid. Use a suitable binary
logit loss. ``masked_prediction`` applies the same configured head to each
non-class token. This binary head returns width one; it cannot supply
8192 pretrained vocabulary logits. The separate multi-dataset MTP checkpoint
has an actual trained 8192-output head. Load it explicitly with
``TFMEncoder.from_pretrained(path, num_classes=8192)``; its
``masked_prediction`` returns ``[batch,electrodes * frames,8192]``.
All vocabulary-head tensors and the learned mask row are retained.

``TFMClassifier`` freezes the complete tokenizer in eval mode and adds a new
linear head over encoder features. Set ``freeze_encoder=True`` for a linear
probe. Both original heads remain in the exported state. Hard IDs stop
raw-input and tokenizer gradients during downstream classification; a
trainable encoder still receives parameter gradients.

Modern execution and independent workflow
-----------------------------------------

Complete official tokenizer outputs, IDs, both input gradients and every
parameter gradient were checked with Torch 2.6 CPU,
``torch.compile(backend='aot_eager', fullgraph=True)``. The complete encoder
was checked with the same backend and ordinary graph breaks permitted;
its tensor-value ID guards preclude fullgraph execution. These checks do
not establish optimized-backend or GPU throughput.

See :doc:`../auto_examples/examples_eegmat_tfm` for the independent real
EEGMat probe. It resamples whole records to 200 Hz, filters 0.1–75 Hz with
a 50-Hz notch, selects the first 16 source electrodes (excluding Fz/Cz/Pz),
and normalizes each four-second window/channel by its 95th absolute percentile
plus 1e-8. That is a teaching transfer protocol; it does not identify those
rows with CHB-MIT's bipolar montage or reproduce CHB-MIT scores.

The last complete subject is held out. Fixed classifier training uses all
remaining subjects without selecting settings on held-out labels. The
example exports both complete frozen states, the new task head, configuration,
source/checkpoint hashes and predictions, then strictly restores and checks
identical predictions. Gallery rendering does not execute this workflow.


Complete multi-dataset MTP pair
-------------------------------

EEG-Arena's default TFM variant uses the author's multi-dataset tokenizer
and MTP-pretrained encoder. This differs from the single-CHB-MIT pair in
both reconstruction and output heads. The complete models keep the same
64-wide, 8192-codebook architecture, with two frequency and two temporal
encoder blocks, eight tokenizer decoder blocks, and four token-encoder blocks.

* ``multiple_dataset_settings/Pretrained_tfm_tokenizer_2x2x8/tfm_tokenizer_last.pth``:
  191 state tensors, MLP Linear(64,64)/Tanh/Linear(64,100), 6881039 bytes;
  SHA256 ``f7929eed0f8275d08a7e18743fdd188c7ba5b4722151333b4d386ca05b84d6d9``.
* ``multiple_dataset_settings/MTP_Pretrained_tfm_encoder_64x4/tfm_encoder_mtp_last.pth``:
  all 59 state tensors including the complete trained 8192-class head,
  5308635 bytes; SHA256
  ``738d56a2021dd363c9d21c4804c441b486b28e33f73883dcf49623f9db8ad973``.

These paths are under the same pinned author's ``pretrained_weigths``
directory. Verify source and exact hashes; no weights are distributed here.
Load the explicit pair and a new downstream head::

    from torcheeg.models import TFMClassifier

    model = TFMClassifier.from_pretrained(
        './tfm_tokenizer_last.pth', './tfm_encoder_mtp_last.pth',
        variant='multi-pretrained', num_classes=2, freeze_encoder=True)

The paired loader defaults to multi-pretrained. Select
``variant='single-chbmit'`` for the old linear/binary pair. Both full heads
remain in the exported state; the downstream classifier uses contextual
class-token features and initializes its own new head. Tokenizer eval mode
prevents codebook EMA and dropout changes, including after ``model.train()``.
The loader rejects mismatched geometry and incomplete states. Verify the
matching pair hashes: tensor shapes alone cannot establish checkpoint identity. Explicit common
``tokenizer_prefix`` and ``encoder_prefix`` support exported key prefixes.
The standalone encoder loader keeps num_classes=1 as its compatible default.

Run the existing independent example with ``--variant multi-pretrained``
and these two local checkpoints. Its default remains single-chbmit for
compatibility. The exported variant, full constructor settings and both
checkpoint hashes identify the chosen pair. Physical channel rows and
preprocessing remain explicit; a multi-dataset pretraining origin does not
make arbitrary montage or signal scaling equivalent.

Both complete components were compared against the unmodified original
classes with the matching actual weights. Eval/train reconstructions, IDs,
input gradients and EMA states matched; full encoder outputs, vocabulary
predictions and every used parameter gradient passed numerical checks,
including ID8192 masking. Independent real-EDF preprocessing, tokenization,
class features and a trained downstream head provide a separate teaching
workflow check. Complete spectral-mask/token-prediction workflows are described below.
Additional released fine-tuning variants require separate validation.


Complementary-spectrum tokenizer pretraining
--------------------------------------------

:class:`torcheeg.models.TFMTokenizerPretrainer` runs the full tokenizer twice
using the original complementary spectral masks. Pass preprocessed floating
EEG ``[batch,electrodes,samples]`` at 200 Hz; the wrapper performs the native
STFT only, without filtering, scaling or resampling. The periodic Hann window
is computed on CPU in float32 and moved to the signal device, preserving the
original source window values even for float64 inputs. Its 200-value
``stft_window`` snapshot is a persistent wrapper buffer. This preserves
exact window values across platform-specific float32 cosine rounding during
checkpoint restore. Supply a fixed ``stft_window`` explicitly for a controlled
reference; a different window is an explicit preprocessing change. Both STFT and temporal
convolution retain only complete 200-sample windows at stride 100.

The default mask drops ten of twenty five-bin frequency groups and
``floor(frames/2)`` temporal positions. Their keep-region intersection is
shared across the entire flattened sample/electrode batch. The second view
is its full complement, not an independent half mask. Each branch receives
the same unmasked waveform, and both reconstruct the ENTIRE clean spectrum.
Do not substitute masked-only MAE loss or mask the raw waveform.

.. code-block:: python

    import torch
    from torcheeg.models import TFMTokenizer, TFMTokenizerPretrainer

    tokenizer = TFMTokenizer.from_pretrained(
        './tfm_tokenizer_last.pth', reconstruction_head='mlp')
    model = TFMTokenizerPretrainer(tokenizer, quantization_mode='native')
    mask_generator = torch.Generator().manual_seed(43)
    output = model(eeg, generator=mask_generator)
    output['loss'].backward()

The output dictionary contains total/reconstruction/quantization losses,
both reconstructions and code-ID arrays, and the Boolean first-view keep
mask. Spectra use ``[batch * electrodes,100,frames]`` and IDs use
``[batch * electrodes,frames]``. In train mode the codebook updates through
EMA twice sequentially, once per branch; the optimizer updates neural
parameters separately. Eval mode leaves all EMA state unchanged.

``quantization_mode='native'`` retains the original sum of codebook and
beta-weighted commitment terms. With straight-through codes and beta=1,
these encoder gradients cancel even though the loss value is nonzero.
``quantization_mode='ema'`` explicitly selects the corrected detached-target
commitment term already provided by TFMTokenizer. It changes the objective
and its gradients; it is not a claim to reproduce the literal source loss.
Save the selected mode with the experiment. Both modes have independent
continuation examples and gradient/resume checks.

The masking generator must be on CPU, matching the source's randperm calls.
Neural dropout uses the ordinary device RNG. An optional explicit Boolean
``mask`` must match the flattened spectrum shape/device and cannot be passed
with a generator. Default masks are batch-shared; an explicit mask is a
user-selected protocol and can deliberately differ.

Symmetric masked-token pretraining
----------------------------------

:class:`torcheeg.models.TFMMaskedTokenPretrainer` requires the complete
matching tokenizer and an encoder with ``num_classes == num_tokens``.
Load the actual trained 8192-output MTP checkpoint; the binary CHB-MIT head
is rejected and no random replacement head is inserted.

.. code-block:: python

    from torcheeg.models import TFMEncoder, TFMMaskedTokenPretrainer

    encoder = TFMEncoder.from_pretrained(
        './tfm_encoder_mtp_last.pth', num_classes=8192)
    model = TFMMaskedTokenPretrainer(tokenizer, encoder)
    output = model(eeg, generator=torch.Generator().manual_seed(43))
    output['loss'].backward()

The wrapper permanently freezes the supplied tokenizer and preserves its
eval mode across ``model.train()``. This prevents the source Lightning
wrapper's initialization-only eval call from allowing later teacher dropout
or EMA updates. Use separate tokenizer instances for tokenizer learning and
frozen-teacher training. Hard IDs stop input and teacher gradients.

For ``[batch,1--16 electrodes,samples]``, clean teacher IDs are
``[batch,electrodes,frames]``. An independent Bernoulli mask keeps each token
in the first view or its complement; ID8192 fills masked positions. Each
view predicts every token using the full vocabulary head, then contributes
mean CE over only its masked positions. The two means are SUMMED, without
partition-size weighting. The output dictionary includes both predictions
``[batch,electrodes * frames,vocabulary]``, component losses, IDs and mask.

``True`` means visible in the first view. Optional ``mask`` must be Boolean
with the teacher-ID shape/device. A supplied generator uses that token
device. Both partitions must be nonempty: the wrapper raises an explicit
error for a degenerate sampled/provided mask instead of the source's NaN
empty-target CE. Advance the generator or provide another mask. At least
two token positions are required. Channel IDs retain TFMEncoder's explicit
sequential-row contract; filtering, units and montage conversion are not inferred.

Training and exact continuation examples
----------------------------------------

Two independent workflows require local complete official multi-dataset
weights and real EEGMat EDF files; neither substitutes a small random model:

* :doc:`../auto_examples/examples_eegmat_tfm_tokenizer_training` continues
  the complete 191-state MLP tokenizer plus the saved window (192 wrapper states), with explicit native or EMA-corrected objective.
* :doc:`../auto_examples/examples_eegmat_tfm_mtp_training` continues the
  complete 250-state teacher/encoder pair plus the saved window (251 wrapper
  states), preserving the trained vocabulary head.

Both use the existing explicit EEGMat transfer preprocessing, separate the
last subject, and train on fixed sequential batches. They save every neural
parameter and EMA buffer, AdamW state, independent mask-generator state,
ordinary dropout RNG, source hashes, preprocessing/constructor/objective
settings and next batch cursor. Restore all of them together. ``--steps``
is the total target optimizer-step count, including resumed steps.
Validation uses a separate fixed generator and performs no update or model
selection. The teaching fixed learning rate differs from the original
large-corpus cosine/warmup schedule; no convergence or benchmark score is claimed.

Independent original-code fixtures cover train/eval losses, both branch
outputs, complete input/parameter gradients and all EMA state. The routine
fixture uses synthetic 16-wide models with nontrivial two-dimensional
attention heads; separate checks use the complete official 64-wide,
8192-codebook, 2+2+8 tokenizer and four-layer MTP encoder. These tests do
not establish GPU/distributed throughput or the paper's training results.


The complete official models also pass PyTorch 2.6 CPU
``torch.compile(backend='aot_eager', fullgraph=False)`` training checks
with explicit masks: both branch outputs, input/parameter gradients and
post-forward EMA states agree with eager execution. Keep ordinary graph
breaks enabled for tensor-value guards. The shared linear-attention projection
and dropout preserve two-dimensional layout before restoring token shape;
this avoids an older AOTAutograd noncontiguous-gradient view failure without
changing any parameter names, dimensions or formula. Independent native
checks and existing BIOT/TFM regressions cover the eager compatibility.
This is not a default-Inductor or GPU performance claim.


EMA state transactions remain outside AOTAutograd graphs, explicitly between
the two compiled neural branches. This preserves both ordered updates and
distributed reductions, while avoiding a PyTorch 2.4 duplicated-mutated-input
compiler error. It does not suppress compiler errors or replace either neural
branch with a smaller network. Training uses graph breaks; inference with
EMA disabled retains the tokenizer's existing compilation behavior.
