CSBrain with an external backend
================================

TorchEEG supports a separately supplied complete original CSBrain backend.
All 295 native state tensors (8,902,000 elements) remain present, including
the complete reconstruction head. No replacement attention network is used.

Source and weights
------------------

See the `paper <https://arxiv.org/abs/2506.23075>`_ and
`author repository <https://github.com/yuchen2199/CSBrain>`_ at
``185aee55b24d0410a830df8dd08d03f675616998``.
The pinned source and linked weight release have **no stated license**.
Obtain them separately under terms appropriate for the intended use.
TorchEEG bundles neither external implementation nor weights.
See ``LICENSES/CSBRAIN-BACKEND-NOTICE.txt`` for provenance.

The complete ``CSBrain.pth`` is 35,722,626 bytes, with SHA256
``cfcdf9dcb069905e5e02e92fe3d59bd3574e779bf554cc02cb701994a79e99cc``.
The author's README links the Google Drive release; its original file ID is
``1Z67je1HQhClyG9XER_zJzFMAKSVNmnF0``.

The explicit example factory verifies the pinned Git revision and unchanged
tracked source. Use a fresh process if another package named ``models`` is
already imported. It restores the exact OMP/BLAS environment after loading
the source, which otherwise changes those variables at import time.
The external source requires ``threadpoolctl`` and ``matplotlib`` in addition
to normal model and dataset dependencies. Library imports load none of it.

Physical input and complete architecture
----------------------------------------

Supply floating EEG ``[batch,electrodes,time]`` at **200 Hz**, with complete
200-sample patches. Time must be a positive multiple of 200. The original
pretraining context contains 30 patches. Shorter contexts are an explicit
transfer choice; no patch repetition or padding is performed.

Declare every physical electrode in input order. The default backend uses:

.. code-block:: text

   FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 FZ CZ PZ

The interface maps case-insensitive names to this original order, then the
native model applies its original frontal/parietal/temporal/occipital/central
permutation. Missing, duplicated or silently renamed electrodes are rejected.
An explicitly constructed different author montage/topology requires both
``backend_channel_names`` and ``allow_montage_transfer=True``. This permits
its original 22-channel BCI fine-tuning geometry without inventing electrodes.
The 19- and 22-channel native paths are independently compared in tests.

The complete model includes all three temporal convolution scales (1, 3, 5),
regional spatial embedding, 200-sample time and 101-bin FFT embeddings,
convolutional position encoding, and 12 width-200 layers. Each layer retains
both original eight-head structured attention branches and its 800-wide
feed-forward network. The final native linear projection remains loaded.
The frozen native mask encoding remains frozen as released.

Features and reconstruction targets
-----------------------------------

``model(eeg)`` returns the mean pre-projection feature ``[B,200]``.
``model(eeg, return_tokens=True)`` returns contextual tokens ``[B,C,P,200]``
in native sorted positions, named by ``model.token_channel_names``.
These contextual positions are not exclusive single-sensor predictions.

The adapter replaces only the projection's parameter-free container with a
view returning its input and original output. Every original child module,
parameter and state key remains intact; the original model forward body
runs unchanged. Consequently the supplied backend is owned by this interface
and its forward now returns that pair. Use the public methods to select it.

``forward_reconstruction(eeg)`` returns the complete native predictions in
**declared target order**. The original trainer compares those output ranks
to pre-sort raw input ranks. Thus target meaning follows the training
criterion, and cannot be inferred from the internal input permutation alone.

``CSBrainClassifier`` adds a new linear task head. With
``freeze_backbone=True``, all original parameters and buffers remain frozen
in evaluation mode while only that head trains. With it disabled, the full
feature backbone can train. This is an explicit transfer head.

Two explicit masked training protocols
--------------------------------------

The original trainer uses the same mask indices for internally sorted input
patches and original-order targets. These are different physical orderings.
For example, native mask index 1 hides **F7**, selects **FP2** as target, and
leaves that FP2 patch visible. This is verified directly on native inputs.
The interface exposes this behavior without silently treating it as physical
sensor masking:

* ``forward_masked(eeg, mask)`` accepts a boolean ``[B,C,P]`` mask in declared
  input order. It maps hiding through the native permutation and compares
  the corresponding physical targets. Returned ``hidden_mask`` and
  ``target_mask`` agree. This is a corrected, **different training protocol**.
* ``forward_native_masked(eeg, native_mask)`` passes the original native
  mask unchanged and reproduces the author's exact criterion. Returned
  ``hidden_mask`` and ``target_mask`` may differ, exposing the mismatch.

Both methods return complete ``prediction``, ``target_mask``, ``hidden_mask``,
``native_mask`` and scalar ``loss``. The first three use declared target/input
order; ``native_mask`` uses internal sorted positions. At least one target
must be selected. Raw target gradients are retained as in the original MSE.

.. code-block:: python

   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   model.train()
   optimizer.zero_grad()
   result = model.forward_masked(eeg, physical_patch_mask)
   result['loss'].backward()
   optimizer.step()

This interface does not establish reproduction of the full original
pretraining dataset, schedule or published benchmark.

Safe loading and restoration
----------------------------

Construct the explicit trusted original backend, then call
``CSBrain.from_pretrained(path, backend, channel_names)``. The complete release
is a flat dictionary with ``module.`` prefixes; these are the defaults.
Complete exports without that prefix use ``prefix=''``. Nested complete
exports can name ``state_key`` explicitly; Safetensors exports also work.
All keys, shapes, dtypes and finite values must match before loading.
There is no unsafe pickle fallback or partial-head loading.

Physical input and backend names/order, sample rate, topology/permutation,
normalization/dropout configuration, and projection view contribute to the
saved fingerprint. Strict restoration validates it even after ``to_empty``.
Keep source/checkpoint hashes, preprocessing, reference, optimizer and RNG
states alongside a training export.

Runnable EEGMat workflow
------------------------

:doc:`../auto_examples/examples_eegmat_csbrain` reads both recordings of each
EEGMat subject using all 19 physical sensors and the original EDF reference.
It resamples whole recordings from 500 to 200 Hz, then applies MNE FIR
0.5--75 Hz and 50 Hz notch filtering. Four-second windows contain 800 samples.
Inputs are microvolts divided by 100; there is no fitted z-score transform.
Background maps to 0 and mental arithmetic to 1.

The last complete subject is held out. Only a new two-class linear head
trains. Every original state, configuration, source/checkpoint hash, test
label and prediction is exported and strictly restored. A real three-subject
run used 120 training windows, 60 held-out windows and 15 head updates.
Its 75% accuracy equals the 75% majority baseline. This small teaching
transfer result is not a published
benchmark score or full original pretraining reproduction. Gallery rendering
does not execute the training example.

Functional tests need no external source. A separate native oracle requires
``TORCHEEG_CSBRAIN_BACKEND_ROOT`` and ``TORCHEEG_CSBRAIN_CHECKPOINT``.
It compares complete native features, reconstructions, both mask conventions,
input and every parameter gradient in evaluation and training modes.
Local documentation success does not establish Read the Docs cloud success.

Real-data masked training and resumption were also checked with two windows:
all 294 trainable native parameters received finite gradients; the released
mask encoding remained frozen. A full model/Adam/RNG export reproduced the
next stochastic update exactly. This checks this training interface, not the
entire original pretraining procedure.

PyTorch 2.6 CPU ``aot_eager`` with ``fullgraph=False`` is checked separately
for features and both masked objectives, including input and every parameter
gradient. Validation requires graph breaks; GPU performance and fullgraph
compilation are not established.
