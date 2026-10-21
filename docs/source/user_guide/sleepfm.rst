SleepFM with external backends
==============================

TorchEEG provides explicit interfaces for three complete separately supplied
SleepFM-Clinical releases: the upstream feature encoder, the five-class
sleep-staging head and the 1065-column diagnosis head. A new classification
head can also train on upstream features. These interfaces retain every
released native state tensor; they do not replace the author networks.

Source, checkpoints and installation
------------------------------------

See the `SleepFM paper <https://doi.org/10.1038/s41591-025-04133-4>`_ and
`author repository <https://github.com/zou-group/sleepfm-clinical>`_.
The source and included weights at revision
``2bcbae04c3592f61352addb7ac3d4193f0a3ca25`` use **CC-BY-NC-4.0**.
Obtain and use those materials separately under their own terms.
See ``LICENSES/SLEEPFM-BACKEND-NOTICE.txt`` for attribution and links.
TorchEEG distributes independent interfaces, validation and metadata handling;
it bundles neither author implementation code nor pretrained weights.

Backend construction is an explicit action using trusted local code. The
library interfaces neither import nor download author source. The runnable
examples below verify the local Git revision and tracked source state, then
explicitly construct the complete author models. They register the loaded
module so PyTorch can locate its classes during compilation. Install the
separately required ``loguru`` dependency in the isolated example environment.

.. code-block:: console

   git clone https://github.com/zou-group/sleepfm-clinical.git ./sleepfm-clinical
   git -C ./sleepfm-clinical checkout 2bcbae04c3592f61352addb7ac3d4193f0a3ca25
   python -m pip install loguru

The three official files reside in that checkout:

.. list-table:: Complete native releases
   :header-rows: 1
   :widths: 30 15 20 35

   * - File under ``sleepfm/checkpoints/``
     - State tensors
     - Tensor elements
     - Native structure
   * - ``model_base/best.pt``
     - 155
     - 4,848,550
     - Convolution tokenizer, spatial attention, six temporal Transformer
       layers, temporal attention pooling and batch-normalization buffers.
   * - ``model_sleep_staging/best.pth``
     - 37
     - 2,335,365
     - Spatial attention, positional encoding, one Transformer layer,
       bidirectional LSTM and all five output classes.
   * - ``model_diagnosis/best.pth``
     - 35
     - 1,792,937
     - Spatial attention, positional encoding, two-layer bidirectional LSTM,
       demographic projection and all 1065 Cox log-risk outputs.

SHA256 values, in that order, are:

* ``ffc9fc10233ebc4d0aae71abce87070db51b8bac6e4b16a5c1b4401a5f73f799``
* ``a742faf4dd9a1f5ebcccc464894081480217e683974750eb3f10e6c4d1b0fde9``
* ``78e4744c5b23e98dbfd779dd523e2f666cdb646db40c6cc56d21060aae3706df``

Loading complete native states
------------------------------

Construct the matching author backend, then call the corresponding
``from_pretrained`` method. The upstream file nests tensors under
``state_dict``; both head files are flat. All three use a ``module.`` key
prefix. Those defaults are explicit in the respective loaders. Other complete
exports can specify ``state_key`` and ``prefix`` without changing tensor
selection. Flat safetensors exports require ``state_key=None`` and the
appropriate explicit prefix.

Every key, shape, dtype and finite value must match. There is no partial
load, unsafe pickle fallback, head removal or positional-table resizing.
Tensor loading uses ``weights_only=True`` on CPU. It does not construct or
execute source code; the caller-supplied backend already exists.

Ordered physical names, modality names and demographic column names are
part of the adapter configuration. Their fingerprint is saved with all native
tensors and checked on strict restoration, including after ``to_empty``.
Save the configuration, source revision, preprocessing and weight hashes
with model states; a weight file does not describe an EEG protocol.

Upstream signal features
------------------------

``SleepFM`` accepts floating ``[batch,electrodes,samples]`` signals sampled
at **128 Hz**, with complete **640-sample, five-second patches** and at most
128 patches per call. The constructor requires ordered physical signal names
and a modality: ``BAS``, ``RESP``, ``EKG`` or ``EMG``. As in the native
embedding-generation workflow, each modality passes separately through the
same complete encoder. Channel names are metadata, not inferred coordinates.

``padding_mask`` is boolean ``[batch,electrodes]`` with ``True`` for absent
padded channels. Each example must retain at least one channel. Temporal
padding is not supported by this upstream call. Filtering, referencing,
units and normalization remain explicit caller responsibilities.

The default output is native pooled features ``[batch,128]``.
``return_tokens=True`` returns contextual features
``[batch,patches,128]`` at five-second resolution. The encoder has temporal
context within each supplied call: arbitrary changes to call boundaries can
change features. The original training workflow uses longer PSG windows;
a short teaching example is a different protocol.

``SleepFMClassifier`` adds a new linear task head. With
``freeze_backbone=True`` the native encoder and buffers remain frozen in
evaluation mode while the new head trains. Otherwise ordinary gradients
can update the complete native encoder.

Sleep staging and padding
-------------------------

``SleepFMStaging`` accepts native embeddings
``[batch,modalities,steps,128]`` and explicit ordered modality names.
The maximum sequence length is 8196. Its output is original five-class
logits ``[batch,steps,5]`` plus a boolean temporal padding mask.

A supplied mask has shape ``[batch,modalities,steps]``. It must separate into
whole-modality padding and a contiguous right suffix of temporal padding.
The first declared modality and first step must be valid for every example.
These restrictions follow the two projections used by native heads to read
spatial and temporal validity. Arbitrary holes or all-missing inputs are
rejected rather than silently interpreted as another mask.

The original staging head has an **unpacked bidirectional LSTM**. Passing
right-padding through that LSTM changes valid predictions through its backward
direction. The adapter calls the complete native head separately on each
actual unpadded sequence, then pads only returned logits with zeros. Tests
compare outputs and every gradient against those original per-sequence calls,
and verify that padded values cannot affect valid predictions or gradients.
Ignore padded output positions in losses and metrics. Stochastic training
follows per-sequence native dropout calls, rather than a padded batch call.

The five native class columns are Wake, N1, N2, N3 and REM. Five-second token
outputs do not automatically define a 30-second annotation protocol. Choose
and record the alignment or aggregation used by a dataset workflow.

Diagnosis outputs and explicit demographics
-------------------------------------------

``SleepFMDiagnosis`` accepts native embeddings
``[batch,modalities,steps,128]``, an explicit mask with the same rules and two
finite, pre-encoded demographic columns ``[batch,2]``. Its maximum sequence
length is 6480. Both demographic column names and order are required.
The model keeps the entire native demographic projection and returns all
**1065 original Cox log-risk columns**, without sigmoid or calibration.
They are not disease probabilities. The original packed LSTM handles valid
sequence lengths; no synthetic right-padding is treated as a true observation.

Obtain the separate official ``sleepfm/configs/label_mapping.csv`` and retain
its complete ``label_idx`` order when naming output columns. Supply the
validated demographic encoding from the matching clinical protocol. The
adapter does not infer that encoding from raw ages or gender descriptions.
The author demonstration PSG, demographics and disease labels are explicitly
synthetic. They establish no real clinical validation or survival performance.

Runnable workflows
------------------

* :doc:`../auto_examples/examples_eegmat_sleepfm` trains a new frozen-encoder
  head on real EEGMat with an unseen-subject holdout. It uses all 19 source
  electrodes, whole-record resampling to 128 Hz, 0.5–45 Hz FIR/50-Hz notch
  filtering and per-window/channel population z-scores. Five-second windows
  are an explicit teaching transfer protocol. The complete model, metadata,
  provenance and predictions are exported and strictly restored.
* :doc:`../auto_examples/examples_sleepedfx_sleepfm_staging` uses real
  Sleep-EDF bipolar EEG with a subject holdout. The reader filters and
  resamples whole records to 128 Hz before extracting scored 30-second
  epochs. Each epoch is an independent six-token sequence, without bridging
  annotation gaps. Only the original final five-class linear layer adapts.
  Six token logits are averaged to match a 30-second label. This short-context
  example is not the published multimodal full-night benchmark.
* :doc:`../auto_examples/examples_sleepfm_diagnosis` performs complete native
  diagnosis inference from an explicit tensor-only contextual-feature export
  and pre-encoded demographics. It validates encoder provenance and all 1065
  disease indices, then saves and strictly restores the complete head and
  outputs. Its input metadata declare whether data are synthetic. It does
  not train on invented survival labels or report clinical metrics.

For a diagnosis input payload, the example docstring specifies required
fields. In its ``metadata_json``, ``feature_encoder`` must be:

.. code-block:: json

   {
     "revision": "2bcbae04c3592f61352addb7ac3d4193f0a3ca25",
     "checkpoint_sha256": "ffc9fc10233ebc4d0aae71abce87070db51b8bac6e4b16a5c1b4401a5f73f799",
     "sampling_rate": 128,
     "patch_samples": 640,
     "representation": "contextual_5s_tokens"
   }

Also provide ordered ``modality_names``, ``demographic_names``, distinct
``record_ids``, actual ``preprocessing`` and ``demographic_encoding``
descriptions and an explicit ``synthetic`` boolean. These provenance checks
do not establish that the caller implemented a clinically valid encoder or
population protocol. Real clinical survival training and validation require
appropriately encoded covariates, event/censoring labels and an independent
study protocol; they are not established by the current numerical examples.

Validation and optional native CI
---------------------------------

Independent interface and workflow tests need no author source. A separate
``SleepFM external backends`` GitHub workflow obtains the pinned, separately
licensed checkout and validates all three checkpoint SHA256 values before
running the complete native oracle. Run that oracle locally with
``TORCHEEG_SLEEPFM_BACKEND_ROOT`` pointing to the explicit pinned checkout.
Without it the three author-backend tests are skipped; interface and workflow
tests remain active. No external download occurs during ordinary imports.

The complete native models were compared in evaluation and stochastic training
modes for outputs, signal/embedding gradients, demographic gradients and every
parameter gradient. Real-data exports were independently checked from raw EDFs
against direct native calls. Torch 2.6 CPU ``aot_eager`` compilation with
permitted graph breaks also preserves outputs and gradients for all three
models. Dynamic masks, sequence lengths and native packed LSTMs require
graph breaks; fullgraph compilation and GPU throughput are not established.
Gallery rendering does not run the training workflows. Local documentation
success does not establish a successful Read the Docs cloud build.

Multi-endpoint survival training
--------------------------------

:class:`~torcheeg.models.SleepFMSurvivalObjective` adds a complete Cox objective
over :class:`~torcheeg.models.SleepFMDiagnosis`. It preserves every native
diagnosis tensor and all 1065 output columns. Supply all ordered endpoint names
explicitly. Follow-up times and binary event indicators are [B,1065] in this
same order. Event time and censoring conventions must be consistent across
subjects. These arrays are measured outcomes, not classification labels. Do not encode
missing follow-up as censoring or fill absent endpoint columns with zeros.

The default ``ties='breslow'`` groups observations with the same time. Every
event uses all subjects with observation time greater than or equal to its
time, including censored subjects at the same time. Contributions are averaged
over observed events within each endpoint, then over all 1065 endpoints.
Zero-event endpoints contribute zero and remain in this final mean. Float64
outputs retain float64 arithmetic; lower precision uses float32 accumulation.
The risk-set definition follows the
`Breslow partial likelihood <https://opensource.nibr.com/torchsurv/_autosummary/torchsurv.loss.cox.html>`_.

For an already constructed complete ``diagnosis`` model and correctly aligned
outcome tensors:

.. code-block:: python

   import torch
   from torcheeg.models import SleepFMSurvivalObjective

   # endpoint_names comes from the ordered outcome schema, not an inferred sort.
   objective = SleepFMSurvivalObjective(diagnosis, endpoint_names)
   optimizer = torch.optim.AdamW(objective.parameters(), lr=0.001)
   optimizer.zero_grad()
   result = objective(embeddings, encoded_demographics, times, is_event,
                      padding_mask)
   result['loss'].backward()
   optimizer.step()

A batch defines its own risk sets. A minibatch loss is not the full cohort
likelihood. Batch composition therefore matters, and a one-subject batch has
zero loss and gradient. Do not duplicate one subject to create a risk set.
The output ``log_risks`` remains [B,1065]. A sigmoid is not a calibrated
survival probability; this interface does not estimate a baseline hazard.

Native reproduction
^^^^^^^^^^^^^^^^^^^

The pinned author pipeline uses ordered prefix risk sets after descending
per-endpoint sorting. It does not group tied times, so equal-time observations
can depend on row order. It also forces float32 log-cumulative accumulation.
This native behavior is separate from the default Breslow objective.

For exact source reproduction, explicitly construct the trusted original
``cox_ph_loss`` callable from the separately supplied source and pass
``ties='native'``, ``native_criterion=author_loss``, and a nonempty
``criterion_id`` identifying its revision and source hash. The callable is
invoked unchanged. TorchEEG does not import or redistribute that training
pipeline. A native criterion must be stateless; source execution remains an
explicit caller action.

Restoration and validation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The complete objective state contains all 35 native diagnosis states, the
physical-metadata fingerprint, and an objective fingerprint. The latter guards
all ordered endpoint names, tie policy, reduction, accumulation policy and
native criterion identity. A changed endpoint order or policy is rejected
during strict restoration. Tensors alone cannot reveal a mislabeled input
column: align input tables against the declared schema before creating tensors.
Demographics must use the original encoded convention; this objective does
not fit or invent their normalization.

For continuation, save the entire objective, optimizer, scheduler, input
cursor and every random state used by the workflow. Construct the trusted
complete backend and matching objective before restoring. The native criterion
code itself is not serialized. Tests compare the full official model, original
loss, input/demographic gradients, every parameter gradient, AdamW updates,
and dropout/RNG/optimizer/scheduler restoration. An independent enumerated
risk-set oracle also checks censoring and ties.

These tests use synthetic inputs and outcomes. The author demo explicitly
states that its PSG, stages, demographics and disease data are synthetic, and
has only one subject row. It is not real clinical survival validation. A
complete standalone caller-cohort workflow and measured clinical outcomes
remain separate from this objective interface.
