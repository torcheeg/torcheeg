User Guide
==========

A TorchEEG workflow starts with a dataset and ends with an evaluation result.
The library components make each intermediate step explicit.

Data and preprocessing
----------------------

A dataset reads source recordings, extracts samples, and stores metadata.
Use :doc:`../torcheeg.datasets` to check the expected file layout, channels,
sampling rate, units, and labels for each source.

Offline transforms run before samples enter the cache. Online transforms run
when samples are read. Label transforms select or map metadata into a target.
See :doc:`../torcheeg.transforms` and :doc:`../torcheeg.io`.
Use a new cache path when source files or offline preprocessing change.
See :doc:`moabb` for sources read through a MOABB paradigm.
See :doc:`cache` for validation, interruption recovery, and legacy-cache limits.

Evaluation partitions
---------------------

Choose partitions before model selection. Keep all windows from a subject or
trial in one group when the protocol requires that isolation. Random sample
splits can place overlapping windows from one recording in different groups.
Use :doc:`../torcheeg.model_selection` to select an appropriate splitter.

For standard holdout evaluation, fit learned preprocessing on training data.
Use validation data to select settings, and reserve test labels for the final
evaluation. State explicitly when a protocol uses unlabeled target data.

For verified runtime combinations and modern API examples, see :doc:`pytorch`.

Models and training
-------------------

Match the sample representation to the model. Raw signals, grid features,
and graphs have different input formats. Check :doc:`../torcheeg.models`
before creating the data loader. Match the output and target formats to
the loss and trainer.

Use :doc:`../torcheeg.trainers` or a standard PyTorch loop.
Start with a small run that checks finite outputs, gradients, and checkpoint
restoration. Then run the full experiment on the intended data.
Record package versions, preprocessing, partition IDs, seeds, and metrics.

Continue with :doc:`../tutorials/index`. The :doc:`../agent_guide` provides
the same library workflow as a reusable assistant skill.

.. toctree::
   :hidden:

   cache
   moabb
   eegpt_decoders
   eegpt_finetuning
   pytorch
   cbramod
   fourier_resampling
   cbramod_heads
   cbramod_tuning
   cbramod_prompts
   cbramod_lora
