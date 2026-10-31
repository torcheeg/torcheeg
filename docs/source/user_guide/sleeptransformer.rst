SleepTransformer
================

:class:`torcheeg.models.SleepTransformer` ports the complete original
`SleepTransformer implementation <https://github.com/pquochuy/SleepTransformer/tree/26167f26ddd8bb59ce787e56655c6e99ba446d2d>`_
from TensorFlow 1 to PyTorch. See the
`paper <https://arxiv.org/abs/2105.11043>`_. Both hierarchical Transformer
stacks, positional encoding, frame-context attention and the full classifier
are retained. The default architecture has 2,448,517 parameters and 65 learned
state tensors. No pretrained weights or TensorFlow runtime are bundled.

Inputs and outputs
------------------

Supply floating spectral input ``[batch, epochs, frames, frequencies, channels]``.
Defaults are 20 epochs, 29 frames, 128 frequency bins and one channel. All
channel/bin orders, physical rates, units and preprocessing are caller-defined.
Use positive context lengths up to the configured maxima. Each context must
contain consecutive epochs from one physical recording; unscored gaps must not
be filled by concatenating unrelated epochs. This API has no padding mask.

Output ``[batch, epochs, num_classes]`` contains categorical logits.
``return_attention=True`` additionally returns normalized frame-context
weights and every frame/sequence self-attention map. These are pre-dropout
probabilities in query/key order. The network observes the full context;
it is not causal, and attention alone does not establish a causal explanation.

There is no extra spectral condenser or multihead output projection. The
original post-normalization feedforward dropout, layer-normalization epsilon
``1e-8`` and learned context pooling are preserved. A generic Transformer
encoder layer adds operations absent from this architecture.

Training objective
------------------

.. code-block:: python

   import torch
   from torcheeg.models import SleepTransformer

   model = SleepTransformer()
   spectral_eeg = torch.randn(2, 20, 29, 128, 1)
   stages = torch.randint(0, 5, (2, 20))
   logits = model(spectral_eeg)
   loss = model.compute_loss(logits, stages)
   loss.backward()

``compute_loss`` reproduces the reference's sum over batch, mean over epoch,
plus ``l2_weight * 0.5 * sum(parameter ** 2)`` over every learned variable,
including biases. This CE is batch-size dependent. Use ordinary flattened
mean cross entropy when reporting per-epoch evaluation CE. Do not add optimizer
weight decay again when reproducing this explicit penalty. PyTorch and
TensorFlow initialization/dropout random streams are not claimed identical.

Checkpoints and complete native verification
-------------------------------------------

Use standard ``state_dict`` for PyTorch checkpoints. ``native_state_dict``
exports all 65 TensorFlow variable names/layouts. ``load_native_state_dict``
accepts those exact names, optionally ending in ``:0``, and explicitly
transposes dense kernels. Strict loading rejects missing, additional, duplicate
or mismatched tensors. Supply model variables only; optimizer slots are not
silently discarded. Obtain any weights from a trusted source and use a safe
numeric representation, rather than executing an untrusted pickle.

The test helper ``test/models/_sleeptransformer_fixture.py`` runs the complete
unchanged author graph at the pinned revision, verifies all five NN/config
file hashes and exports all variables, outputs, original losses, input/all
parameter gradients and one full SGD update. It requires an isolated Python
3.7 / TensorFlow 1.15.5 environment. TensorFlow 2 compatibility substitutes or
reduced neural modules are not used. For example, with the complete source
checked out separately::

   python test/models/_sleeptransformer_fixture.py \
       --reference /path/to/SleepTransformer --output /tmp/sleeptransformer.npz

Then, from the TorchEEG validation environment::

   TORCHEEG_SLEEPTRANSFORMER_FIXTURE=/tmp/sleeptransformer.npz \
       python -m unittest test.models.test_sleeptransformer \
       test.models.test_sleeptransformer_native \
       test.examples.test_sleepedf_sleeptransformer

Cross-framework arithmetic checks disable dropout while retaining every NN
layer and parameter. Independent tests cover PyTorch dropout, training RNG and
optimizer next-step restoration, physical PSD equations, finite differences,
batch isolation and strict checkpoints. The workflow creates the oracle in CI;
the exported numerical fixture is not distributed as a pretrained model.

Real-data teaching workflow
---------------------------

See :doc:`../auto_examples/examples_sleepedf_sleeptransformer`. It uses
Sleep-EDF Fpz-Cz at 100 Hz, explicit 0.3--40 Hz whole-record filtering and
30-second epochs. Symmetric Hamming windows of 200 samples, hop 100 and FFT
256 give 29 frames; dropping DC leaves 128 positive bins. Compute the
one-sided microvolt-squared/Hz PSD and log it with a ``1e-12`` floor.

Choose fixed, nonoverlapping contiguous 20-epoch contexts without stage-based
selection. Hold out the final subject including all nights. Fit per-bin/channel
statistics only on training subjects. W/N1/N2/N3/REM map to 0/1/2/3/4. The
checkpoint includes complete model/optimizer/RNG, training statistics, physical
identities, hashes and held-out predictions, with atomic no-overwrite export.
This short teaching protocol is not the paper's SHHS protocol or an Arena
score. Full-record filtering and full-context inference are offline operations.

Provenance
----------

Preserve the original MIT copyright/permission in
``LICENSES/SLEEPTRANSFORMER-LICENSE.txt``. The reference credits Transformer
blocks to Kyubyong Park; retain their Apache-2.0 license in
``LICENSES/SLEEPTRANSFORMER-TRANSFORMER-LICENSE.txt``. The PyTorch port changes
framework and adds explicit public shapes, checkpoint conversion and boundary
validation. No raw EEG or author weights are redistributed.
