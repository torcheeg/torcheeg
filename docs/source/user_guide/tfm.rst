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

This integration targets the paired **single-dataset CHB-MIT** release in
``pretrained_weigths/single_dataset_settings/CHBMIT_tfm_tokenizer_2x2x8/``.
It does not establish support for every multi-dataset or fine-tuned release.

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
The released encoder has a **single-logit binary classification head**;
``forward`` returns ``[batch,1]`` without sigmoid. Use a suitable binary
logit loss. ``masked_prediction`` applies the same configured head to each
non-class token. The released head therefore returns width one, not 8192
pretrained vocabulary logits. Constructors permit a separately trained
vocabulary head; it cannot be inferred from this binary checkpoint.

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
