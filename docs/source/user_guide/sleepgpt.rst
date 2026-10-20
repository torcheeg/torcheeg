SleepGPT with an external backend
=================================

TorchEEG provides an explicit interface for the complete released SleepGPT
large time-frequency model. It retains both reconstruction heads and all
257 native state tensors (137,257,856 elements). Feature extraction, fine
tuning and native masked reconstruction use the complete author network.

Source, weights and installation
--------------------------------

Use the `author repository <https://github.com/LordXX505/SleepGPT>`_ at
``fcbb2c7a6489b5742ab77d66483661e45a4aadf2`` and the
`Figshare checkpoint release <https://doi.org/10.6084/m9.figshare.30626870.v1>`_.
The pinned source has **no stated license**. The separate weight release
specifies **CC BY 4.0**; this does not establish source redistribution rights.
Obtain the materials separately under their own terms. TorchEEG bundles
neither author code nor weights. See ``LICENSES/SLEEPGPT-BACKEND-NOTICE.txt``.

The published file is ``ModelCheckpoint-epoch=79-val_acc=0.0000-val_score=4.2305.ckpt``
(Figshare file 59576246, 1,647,356,391 bytes). Its SHA256 is
``a30c53422eaea0bdcd6a1064505bc4ab27a4e4c5bbb6f47ba602e2e10021ed83``
and MD5 is ``c50548d45b6480701364dee0653b3eec``.

Use an isolated environment with the normal model and dataset dependencies,
PyTorch 2.6, ``lightning==2.6.6``, ``timm``, ``transformers>=4,<5`` and
``nvidia-ml-py``. The explicit example factory checks the source Git revision
and tracked modifications, then imports the original model and helper bodies
through their package paths. It leaves unrelated dataset/trainer package
entrypoints unexecuted. Use a fresh process if another package named ``main``
is already loaded. Library imports do not import or download this backend.

The factory disables the author's legacy checkpoint loader and uses the
restricted TorchEEG loader instead. It supplies only the documented later
source defaults ``visual=False`` and zero Spindle loss. The released model
has no pooler parameters; a parameter-free ``Identity`` exposes its native
features. Both reconstruction heads remain present. The checkpoint's
``use_cb=False``, drop-path rate 0.1 and L1 reconstruction losses are retained.

Physical input and representations
-----------------------------------

``SleepGPT`` requires floating signals ``[batch,electrodes,3000]`` at
**100 Hz**, representing exactly **30 seconds**. Its complete network has
12 layers, width 768, 16 attention heads and fifteen 200-sample patches
per sensor. Ordered channel names must come from the original vocabulary:
``C3, C4, EMG, EOG, F3, Fpz, O1, Pz``. Unknown names, duplicates and automatic
bipolar-to-monopolar renaming are rejected. Available channels are mapped to
their original slots. Absent native sensors are zero-filled and masked.

An optional boolean ``padding_mask[batch,electrodes]`` uses ``True`` for
absent input sensors; every example needs an available sensor. Arbitrary
values at absent positions cannot change valid features or input gradients.
The caller must supply the correct reference, units, filtering and scaling.
The model does not resample, truncate, repeat or time-pad inputs implicitly.

The spectral branch uses the native detached standardized STFT: 256-point
FFT, periodic 200-sample Hann window, hop 100, reflection centering, the first
100 frequency bins and all frames after the first. Native 10-log magnitude
values are normalized across frequencies using sample standard deviation.
Zero-variance frames become zero. The implementation uses current complex
STFT output and matches the original real/imaginary implementation. As in
native pretraining, the spectral preprocessing is detached from raw EEG
gradients; spectral encoder and decoder parameters still receive gradients.

``forward`` returns the concatenation of both native class features,
``[batch,1536]``. ``forward_tokens`` returns both contextual token grids
``[batch,8,15,768]`` and a canonical sensor padding mask. Padded query tokens
can contain context; exclude them from downstream token reductions.
``SleepGPTClassifier`` adds a new linear task head. With
``freeze_backbone=True`` the full native model stays in evaluation mode
while only that new head trains. Otherwise all native parameters can adapt.

Native masked reconstruction
-----------------------------

``forward_masked`` accepts an explicit boolean time mask
``[batch,input_electrodes,15]``: ``True`` selects a reconstruction target.
Each example needs at least one selected available patch. Selecting absent
sensors is rejected. Frequency masking follows the original stochastic
sampler and ordinary PyTorch RNG; it is not silently replaced by the time
mask. Empty native frequency-loss selections are rejected.

The result includes both original reconstruction predictions
``[batch,120,200]``, canonical masks ``[batch,8,15]``, the native L1
``time_loss`` and ``frequency_loss``, their sum ``loss``, and sensor padding.
For a constructed complete model, an ordinary training step is:

.. code-block:: python

   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   model.train()
   optimizer.zero_grad()
   result = model.forward_masked(eeg, time_mask, padding_mask=padding_mask)
   result['loss'].backward()
   optimizer.step()

Keep the original target-gradient convention; the adapter does not detach
raw time targets. Save optimizer and RNG state when resuming stochastic
pretraining. This interface is not a reproduction of the full published
PSG pretraining schedule or population normalization.

Complete loading and restoration
---------------------------------

Construct the explicit trusted local backend, then use
``SleepGPT.from_pretrained(path, backend, channel_names)``. The default
``state_key='state_dict'`` matches the Lightning release. All keys, shapes,
dtypes and finite tensor values must match before loading. There is no
unsafe pickle fallback, partial load, decoder removal or positional resizing.
Loading uses ``weights_only=True`` on CPU with memory mapping. Flat
safetensors exports can use ``state_key=None``. An explicit ``prefix`` only
selects a complete namespace; it does not permit missing native tensors.

The adapter saves a physical/configuration fingerprint alongside every native
state. Strict restoration checks ordered sensor names and behavioral settings,
including L1 loss and drop-path configuration, even after ``to_empty``.
Save source revision, checkpoint hash, preprocessing and input reference with
the model. States alone cannot specify a complete EEG protocol.

Runnable workflow and verification
-----------------------------------

:doc:`../auto_examples/examples_eegmat_sleepgpt` reads real EEGMat recordings,
holds out a complete subject and trains a new linear head on frozen complete
SleepGPT features. It resamples whole recordings from 500 to 100 Hz, applies
0.5--40 Hz FIR filtering and selects physical C3, C4, F3, O1 and Pz in their
original reference. It omits a 50 Hz notch at the 100 Hz Nyquist boundary.
Per-epoch/channel population z-scores use a 1e-6 microvolt standard-deviation
floor. Missing EMG, EOG and Fpz are explicitly masked. This teaching transfer
protocol differs from the original PSG normalization and published benchmark.

The example exports and strictly restores all 260 adapter/classifier states,
physical metadata, source/checkpoint hashes and held-out predictions.
Its real-data run was independently checked from raw EDFs using direct native
spectral preprocessing and complete author inference. A short frozen probe
establishes runnable transfer, not competitive sleep decoding performance.

Independent interface and workflow tests do not need author source.
The optional ``SleepGPT external backend`` CI obtains the pinned checkout
and validates the complete weight hash before comparing native outputs,
both reconstruction losses, raw EEG gradients and every parameter gradient
in evaluation and stochastic training modes. Run that oracle with
``TORCHEEG_SLEEPGPT_BACKEND_ROOT`` and ``TORCHEEG_SLEEPGPT_CHECKPOINT``.
Without those paths, only the separate native oracle is skipped.

PyTorch 2.6 CPU ``aot_eager`` compilation with permitted graph breaks preserves
feature/reconstruction outputs and all gradients. The author code can reach
Dynamo's recompilation cache limit and fall back to eager execution; this is
not fullgraph compilation or a performance claim. Gallery rendering does not
run training examples. Local builds do not establish Read the Docs success.
