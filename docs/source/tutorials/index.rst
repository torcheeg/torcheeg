Tutorials
=========

Follow the components in order for an introduction to TorchEEG.
Some examples need separately obtained data. Check a script's data paths
and model parameters before running it.

.. toctree::
   :maxdepth: 1

   Intro <../auto_examples/examples_pipeline>
   Dataset <../auto_examples/examples_datasets>
   Transformation <../auto_examples/examples_transforms>
   Partition <../auto_examples/examples_model_selection>
   Model <../auto_examples/examples_models>
   Trainer <../auto_examples/examples_trainers>
   EEGPT input adapters <eegpt_input>
   All examples <../auto_examples/index>

For a workflow built directly with PyTorch, see
:doc:`../auto_examples/examples_vanilla_torch`.
A documentation build does not execute these training scripts. A successful
build therefore does not establish that an experiment has run.

Generative modeling
-------------------

:doc:`../auto_examples/examples_eegmat_bvae` trains an unconditional VAE
with the public EEGMat dataset, holds out a subject, and saves the model
and generated differential-entropy feature grids. It uses the current
:class:`torcheeg.trainers.BetaVAETrainer` API without a search service or
a pretrained image metric. The example explains its signal units,
preprocessing, split and diagnostic metrics.

Motor imagery with a shallow CNN
--------------------------------

:doc:`../auto_examples/examples_bciciv2b_shallowconvnet` trains
:class:`torcheeg.models.ShallowConvNet` with a held-out BCI Competition IV 2b
subject. It uses explicit EEG-Arena architecture settings and a fixed local
training workflow. The preprocessing and one-epoch validation run do not
reproduce the complete EEG-Arena benchmark protocol. The model reference
also documents the classic unconstrained paper configuration.

Motor imagery with a four-block CNN
-----------------------------------

:doc:`../auto_examples/examples_bciciv2b_deepconvnet` trains
:class:`torcheeg.models.DeepConvNet` with the same held-out subject workflow.
It states the constrained architecture settings and enables kernel adaptation
explicitly. The model reference documents the fixed-kernel classic configuration
and the effective kernel used for shorter windows.

EEGPT encoder features and a linear task head
---------------------------------------------

:doc:`../auto_examples/examples_bciciv2b_eegpt` states channel IDs,
sampling rate, tensor shapes and the subject holdout. Its default small
encoder starts from random weights. Optional complete local encoder states
are loaded with strict key/shape checks and safe tensor deserialization.
An executed random-initialization example is not pretrained validation.
The encoder API also supports electrode/time masks and attention inspection.

EEGPT classification, regression and feature extraction
-------------------------------------------------------

The direct summary-token wrappers take a configured encoder. They do not
resample signals or replace electrode identities with a learned channel mix.
The task heads start from random weights even when an encoder checkpoint
is loaded. Freezing an encoder keeps it in evaluation mode during head
training. Save a full task state to restore trained task-head weights.

* :doc:`../auto_examples/examples_bciciv2b_eegpt_classifier` trains
  :class:`torcheeg.models.EEGPTClassifier` with a subject holdout.
* :doc:`../auto_examples/examples_eegmat_eegpt_regression` uses
  :class:`torcheeg.models.EEGPTRegression` to predict the public EEGMat
  subtraction-score metadata, with all windows of each subject kept together.
* :doc:`../auto_examples/examples_bciciv2b_eegpt_embeddings` saves
  :class:`torcheeg.models.EEGPTEmbedding` features and aligned metadata.
  Fit any later feature scaler or task model on a training-subject split.

Each example states its actual source, preprocessing and weight status. The
small random-encoder defaults are functional demonstrations. They do not
establish pretrained EEGPT accuracy or reproduce the full published
predictor/reconstructor evaluation architecture.
