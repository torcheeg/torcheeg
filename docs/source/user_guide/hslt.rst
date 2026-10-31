HSLT with complete hierarchical spatial learning
================================================

:class:`torcheeg.models.HSLT` implements the complete MIT-licensed
`LibEER HSLT <https://github.com/XJTU-EEG/LibEER>`_ at revision
``dddff9776dbdae21195fe320dff0a5ba61628a18``. LibEER independently
reimplements Wang et al.'s hierarchical spatial learning Transformer
(`DOI:10.1109/JSEN.2022.3144317 <https://doi.org/10.1109/JSEN.2022.3144317>`_).
This does not establish numerical equivalence to unavailable original-author
code. It is a supervised feature model, rather than a foundation model.

Complete architecture and defaults
----------------------------------

Each of nine regions has a learned electrode embedding/class token and
position embeddings, followed by two electrode Transformer blocks. Regions
with unequal electrode counts retain their full learned token projections.
Concatenated regional representations are embedded into a region sequence
with its own class token and positions, followed by two region Transformer
blocks and the complete linear classifier.

The native electrode and region embedding widths are eight and sixteen.
Each attention block projects through ReLU to width 64 and sixteen heads,
then projects back with ReLU. Pre-normalized residuals and both GELU/dropout
MLP stages are preserved. The default complete model has 373,330 parameters
and 368 state tensors. Every state key matches the pinned reference.

Public parameter names use explicit electrode/region embedding widths and
depths instead of the reference's ``De/Dr/Le/Lr`` abbreviations. Constructor
arguments are authoritative: the library does not read a current-directory
YAML file or print settings as a side effect. It does not depend on LibEER
at runtime and does not bundle or automatically obtain pretrained weights.
The full MIT copyright and permission notice is distributed with the package.

Grouped feature input
---------------------

Input is floating ``[batch, num_electrodes, in_channels]``, where
``in_channels`` is the number of features **per electrode**, such as band
differential entropy. It is not a temporal raw EEG axis. Only the complete
32-row DEAP and 62-row SEED regional geometries are supported.

Electrodes must already be concatenated in the following order. No implicit
channel selection, interpolation, reordering or feature computation occurs.

.. list-table:: DEAP regional order
   :header-rows: 1

   * - Group
     - Ordered electrodes
   * - PF
     - FP1, AF3, AF4, FP2
   * - F
     - F7, F3, FZ, F4, F8
   * - LT
     - FC5, T7, CP5
   * - C
     - FC1, C3, CZ, C4, FC2
   * - RT
     - FC6, T8, CP6
   * - LP
     - P7, P3, PO3
   * - P
     - CP1, PZ, CP2
   * - RP
     - P8, P4, PO4
   * - O
     - O1, OZ, O2

.. list-table:: SEED order from the pinned reference's transfer mapping
   :header-rows: 1

   * - Group
     - Ordered electrodes
   * - PF
     - FP1, FPZ, FP2, AF3, AF4
   * - F
     - F7, F5, F3, F1, FZ, F2, F4, F6, F8
   * - LT
     - FT7, FC5, FC3, T7, C5, C1
   * - RT
     - FT8, FC4, FC6, T8, C2, C6, CP6
   * - C
     - FC1, C3, CZ, FCZ, FC2, C4
   * - LP
     - TP7, CP5, CP3, P7, P5, P3, P1, PO3
   * - P
     - CP1, CP2, CPZ, PZ
   * - RP
     - TP8, CP4, P8, P6, P2, P4, PO4
   * - O
     - PO7, PO5, POZ, PO6, PO8, CB1, O1, O2, OZ, CB2

For SEED the reference's unused region display names put C before RT,
whereas its transfer mapping and actual seven/six token counts put RT before
C. The table specifies the actual mapping and counts; no neural operation
is changed to silently reinterpret that reference inconsistency.

Native probabilities or explicit logits
---------------------------------------

By default the complete native activation is retained: two-class output
uses two independent sigmoids, and other class counts use softmax. Two
sigmoid values need not sum to one. Do not treat them as normalized
categorical probabilities. To use the normal TorchEEG classifier or cross
entropy on categorical labels, explicitly request pre-activation logits:

.. code-block:: python

   import torch
   from torcheeg.models import HSLT

   model = HSLT(in_channels=4, return_logits=True)
   grouped_features = torch.randn(2, 32, 4)
   logits = model(grouped_features)
   loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
   loss.backward()

The optional logits output retains the identical classifier and every
learned state. Native sigmoid output can instead be paired with BCE and
explicit two-column targets; softmax output can be paired with NLL of its
logarithm. Store the output convention with model settings in checkpoints.
The native attention implementation requires batch-first attention support;
native validation uses PyTorch 2.6. Historical framework binaries are not
claimed solely from modern tests.

Independent DEAP workflow
-------------------------

The :doc:`DEAP example </auto_examples/examples_deap_hslt>` uses trusted
provider preprocessed Python files: 32 EEG channels, microvolts, 128 Hz,
provider 4--45 Hz filtering. It takes the first six stimulus seconds per
trial, excluding the preceding three-second baseline, without baseline
subtraction. It computes four fifth-order causal Butterworth differential
entropy bands (4--8, 8--14, 14--31, 31--45 Hz), explicitly reorders electrodes
and fits each electrode/band mean and population standard deviation on
training subjects only. The held-out subject never fits normalization.
Valence at least five is class one, below five class zero.

Fixed epochs do not select checkpoints on held-out labels. The export is
atomic and refuses overwrite; it contains full model/optimizer/random states,
training feature statistics, physical/channel/label protocol, source hashes
and test logits/targets. CLI resume is not implemented. Four bands and one
window per trial are explicit teaching choices; this example does not
reproduce a paper's five-band experiment or EEG-Arena's raw-signal adaptation.

Validation scope
----------------

Native checks import the complete pinned LibEER source and constants, not a
substitute neural network. They compare default/custom initialization and
RNG, every state, both supported geometries, train/eval native outputs,
BCE/NLL, input and all parameter gradients and AdamW updates. Independent
embedding/attention arithmetic, strict state loading, sample isolation,
logits/activation consistency, full next-step optimizer/RNG restoration,
eager full-graph compilation and the runnable example are also tested.
Source-dependent tests explicitly skip when the separate reference is absent;
dedicated CI obtains and verifies it. Synthetic checks and short real-data
runs do not establish paper scores. GPU, AMP, optimized compilation and
distributed execution are not claimed.
