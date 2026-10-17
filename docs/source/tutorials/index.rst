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
