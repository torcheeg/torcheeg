Foundation models
=================

Choose a backbone and read its input contract before loading a checkpoint.
EEG models differ in sampling rate, signal units, electrode order, reference,
window length, normalization, and patch layout. A matching tensor shape alone
does not establish a valid input.

Available interfaces
--------------------

Import public interfaces from ``torcheeg.models``. The guide for each family
describes its supported variants, trusted checkpoint source, preprocessing,
standalone examples, and validation limits. A downstream classifier does not
imply that a complete pretraining objective is available.

.. list-table:: Model families
   :header-rows: 1
   :widths: 20 35 45

   * - Family
     - Backbone
     - Usage
   * - BIOT
     - :class:`~torcheeg.models.BIOT`
     - :doc:`biot`
   * - BENDR
     - :class:`~torcheeg.models.BENDR`
     - :doc:`bendr`
   * - EEGPT
     - :class:`~torcheeg.models.EEGPT`
     - :doc:`eegpt_decoders`
   * - LaBraM
     - :class:`~torcheeg.models.LaBraM`
     - :doc:`../torcheeg.models`
   * - NeuroGPT
     - :class:`~torcheeg.models.NeuroGPT`
     - :doc:`neurogpt`
   * - CBraMod
     - :class:`~torcheeg.models.CBraMod`
     - :doc:`cbramod`
   * - CSBrain
     - :class:`~torcheeg.models.CSBrain`
     - :doc:`csbrain`
   * - EEGMamba
     - :class:`~torcheeg.models.EEGMamba`
     - :doc:`eegmamba`
   * - LEAD
     - :class:`~torcheeg.models.LEAD`
     - :doc:`lead`
   * - LUNA
     - :class:`~torcheeg.models.LUNA`
     - :doc:`luna`
   * - mdJPT
     - :class:`~torcheeg.models.MDJPT`
     - :doc:`mdjpt`
   * - NeuroRVQ
     - :class:`~torcheeg.models.NeuroRVQ`
     - :doc:`neurorvq`
   * - REVE
     - :class:`~torcheeg.models.REVE`
     - :doc:`reve`
   * - CodeBrain
     - :class:`~torcheeg.models.CodeBrain`
     - :doc:`codebrain`
   * - ST-EEGFormer
     - :class:`~torcheeg.models.STEEGFormer`
     - :doc:`steegformer`
   * - SleepFM
     - :class:`~torcheeg.models.SleepFM`
     - :doc:`sleepfm`
   * - SleepGPT
     - :class:`~torcheeg.models.SleepGPT`
     - :doc:`sleepgpt`
   * - TFM
     - :class:`~torcheeg.models.TFMEncoder`
     - :doc:`tfm`
   * - Uni-DMFM
     - :class:`~torcheeg.models.UniDMFM`
     - :doc:`unidmfm`

HEAR base and large are not implemented. The complete trusted author source
and official checkpoints have not been obtained. This entry does not count
as pretrained-model support.

Load a trusted checkpoint
-------------------------

Use the checkpoint loader documented for the selected family. Preserve its
complete architecture and required metadata. Do not substitute a randomly
initialized model or load a partial state and report it as pretrained.

Some families use an explicitly supplied author backend. Importing TorchEEG
does not download or execute that external code. Read the guide, check the
pinned revision and license, then construct the backend explicitly. The
TorchEEG package does not redistribute those source trees or weights.
Separate source and weight licenses can impose different conditions.

Keep electrode names and order, units, sampling rate, preprocessing, source
revision, and checkpoint identity with exported states. Runtime geometry is
part of the model contract even when an author checkpoint omits it. Restore
through the documented API and verify the result on the same input.

Self-supervised training
------------------------

The following interfaces expose complete reconstruction or masked-token
objectives. Read the linked guide for the exact loss, masking, codebook
updates, and the distinction between an author protocol and a teaching
workflow.

* :class:`~torcheeg.models.CodeBrainTokenizer`: :doc:`codebrain_tokenizer`.
* :class:`~torcheeg.models.STEEGFormerPretrainer`: :doc:`steegformer`.
* :class:`~torcheeg.models.TFMTokenizerPretrainer` and
  :class:`~torcheeg.models.TFMMaskedTokenPretrainer`: :doc:`tfm`.
* :class:`~torcheeg.models.NeuroRVQTokenizer` and
  :class:`~torcheeg.models.NeuroRVQMaskedTokenPretrainer`: :doc:`neurorvq`.
* :class:`~torcheeg.models.UniDMFMPretrainer`: :doc:`unidmfm`.

Other interfaces also expose their native reconstruction or paired objectives
where documented. A short example that updates a full model verifies that
workflow; it does not reproduce the paper's pretraining corpus or benchmark.

Evaluate on a separate subject or recording group. Keep the final test labels
out of checkpoint and parameter selection. Exact continuation requires the
model, optimizer, scheduler when used, data cursor, preprocessing identity,
and every random state used by augmentation or masking.

Validation and scope
--------------------

Functional tests are under ``test/models``. Complete external-source numerical
tests require explicit trusted checkouts and official checkpoints; their
absence is reported as a skip. Those checks compare complete native outputs
and gradients. Portable synthetic fixtures test equations and regressions,
and do not contain released checkpoint weights or patient recordings.

See :doc:`pytorch` for supported runtime checks. CPU compilation evidence does
not establish GPU, mixed-precision, distributed, or performance behavior.
Use the standalone example linked in the selected guide and record its data
requirements before running a larger experiment.
