BiDANN with complete hemisphere and domain-adversarial paths
============================================================

:class:`torcheeg.models.BiDANN` adapts the complete MIT-licensed
`LibEER BiDANN <https://github.com/XJTU-EEG/LibEER>`_ at revision
``dddff9776dbdae21195fe320dff0a5ba61628a18``. It is an independent
reimplementation of Li et al.'s bi-hemisphere domain-adversarial EEG emotion
model, not an original-author numerical equivalence claim or foundation model.
The full MIT notice is distributed with the adaptation. LibEER itself is not
a runtime dependency and no pretrained weights are bundled or downloaded.

Complete neural architecture
----------------------------

Both hemisphere paths retain their own two-layer, 128-hidden-unit LSTM with
native dropout and initialization. Each learns a temporal projection to eight
positions and retains BN/ReLU. The concatenated features feed the complete
emotion classifier. Left and right local domain discriminators and the global
domain discriminator retain every layer, BN buffer and gradient reversal.
The native default has 3,348,649 parameters and 94 state tensors.

The LSTMs are separate hemisphere encoders; they are not silently replaced
with a single bidirectional LSTM. Public ``sequence_length`` and
``adversarial_weight`` give explicit names to the reference's length and
gradient-reversal arguments. Requested ``num_classes`` is honored, correcting
the reference's hardcoded three-class classifier. The default three-class
network preserves every native state and initialization RNG.

Ordered feature sequences and hemisphere layouts
------------------------------------------------

Provide floating ``[batch, sequence_length, 62, in_channels]`` in
``SEED_CHANNEL_LIST`` order. ``in_channels`` means features per electrode per
frame, normally five provider DE bands; it is not a raw temporal sample axis.
The caller supplies physical feature rates, units, preprocessing and channel
order. Only the complete 62-row SEED hemisphere mapping is supported; no
implicit channel interpolation, filtering or feature extraction occurs.

Hemisphere indexing is differentiable and stays on the input device. The
reference converted input to NumPy/pandas and detached its gradients. The
library retains all neural layers while allowing gradients through source
and target input features.

The reference also transposes the right hemisphere from ``B,T,F,C`` to
``B,F,T,C`` and reshapes it back to ``B,T,-1``. This mixes temporal and feature
positions. The default ``hemisphere_layout='aligned'`` corrects that data
layout: each time frame keeps its ordered electrode/feature pairs. Both
training and inference use the chosen layout consistently.

For strict replay of a complete LibEER checkpoint, explicitly use
``hemisphere_layout='libeer'``. This preserves the reference's literal right
reshape and matches its complete original forward calculations. The layout
flag changes no parameter shapes or state keys; it must be saved with model
settings because the same weights have different input meanings under the
two layouts. Corrected alignment is validated against independent NumPy
indexing and every full unmodified reference neural component, not claimed
as bug-for-bug equivalence to the original data helper.

Classification and full adversarial training
--------------------------------------------

Calling ``model(source)`` returns ordinary categorical logits through both
hemisphere encoders and the complete classifier. This works for inference or
an explicitly chosen classification-only workflow. Calling
``model(source, target)`` returns the full native tuple:
``(source_logits, left_domain_logits, right_domain_logits, global_domain_logits)``.
Source and target minibatches may have different sizes. Every domain head
uses concatenated source rows followed by target rows.

.. code-block:: python

   import torch
   from torcheeg.models import BiDANN, BiDANNLoss

   model = BiDANN()
   source = torch.randn(3, 9, 62, 5)
   target = torch.randn(2, 9, 62, 5)
   outputs = model(source, target)
   loss = BiDANNLoss()(outputs, torch.tensor([0, 1, 2]))
   loss.backward()

:class:`torcheeg.models.BiDANNLoss` computes source CE plus the sum of all
three domain CEs with native domain labels (source zero, target one).
``domain_weight`` defaults to one. The model already reverses discriminator
gradients into its encoders with ``adversarial_weight``; the loss does not
reverse them a second time. No target emotion label is accepted by the loss.
Both scales are finite and nonnegative. Training BN in the classifier needs
at least two source rows, as in the reference.

The native misspelled ``classifer.`` state prefix is intentionally preserved
for strict checkpoint loading; the read-only ``classifier`` property exposes
the same complete head without registering duplicate parameters. Save the
full model, optimizer, random states, geometry, layout and source/target
protocol. Calling a classification-only path does not train domain heads.

Independent real-data UDA workflow
----------------------------------

The :doc:`SEED example </auto_examples/examples_seed_bidann>` reads provider
``de_LDS`` feature MAT files and ``label.mat``. It uses the first nine
consecutive one-second frames within each subject/date/session/trial, with
62 ordered electrodes and five provider bands. Original EEG was sampled at
200 Hz; the input to this model is one feature frame per second. Provider
filtering and LDS smoothing are used as supplied rather than recomputed.

The last sorted subject supplies unlabeled target-domain features during
training. Its emotion labels are used only for final evaluation, not the
loss or checkpoint selection. This is **transductive unsupervised domain
adaptation**, not a standard holdout where target features were never seen.
Input feature mean/std is fitted on source subjects only. Model BN updates
on both domains during adaptation, intentionally as part of the algorithm.

Emotion -1/0/1 maps to 0/1/2. Fixed-epoch training uses the complete four-head
loss, cycles unlabeled target minibatches and retains every source sequence.
Select a minibatch size without a one-source-row remainder for native BN.
The atomic no-overwrite export saves complete model/optimizer/random states,
source-fitted feature statistics, target outputs and all source/target,
physical/channel/label/layout/provenance metadata. CLI resume is not
implemented. This short functional example does not reproduce a paper score,
full sequence protocol or all fixed EEG-Arena variants/tasks.

Validation scope
----------------

Tests import the complete pinned LibEER source and training formula. They
compare default/custom initialization/RNG and every state, full original
legacy train/eval outputs and all four losses, all parameter gradients, BN
and AdamW updates. Corrected alignment is separately compared against
independent indexing and all complete reference neural paths/gradients/
updates. Independent gradient-reversal direction, source-input finite
differences, requested class counts, source-only inference, sample isolation,
strict state loading and complete optimizer/RNG next-step resume are tested.

Eager compilation is checked with the native LSTM graph breaks permitted;
full-graph or optimized compilation is not claimed. Native-dependent tests
skip explicitly when the full separate source is absent; dedicated CI
obtains and verifies it. GPU, AMP, distributed execution and original-author
or fixed-Arena numerical/protocol equivalence are not claimed.
