CBraMod low-rank tuning
=======================

:func:`torcheeg.models.add_cbramod_lora` adds trainable low-rank updates to
explicit projections of a CBraMod backbone or task model. Load a complete
local backbone checkpoint first when pretrained weights are needed. Install
LoRA before constructing the optimizer and before restoring augmented state.
The helper downloads no data or weights.

Projection names and mathematics
--------------------------------

Each selected projection computes::

    x W^T + b + (alpha / rank) * dropout(x) A B

A has shape ``[in_features, rank]`` and B has shape
``[rank, out_features]``. A uses Kaiming initialization and B starts at zero.
The default targets are ``('linear1', 'linear2')``, retaining the archived
workspace's feed-forward selection. These explicit attention targets are
also supported:

* ``self_attn_s.q_proj``, ``self_attn_s.k_proj``, ``self_attn_s.v_proj`` and
  ``self_attn_s.out_proj`` for spatial attention.
* ``self_attn_t.q_proj``, ``self_attn_t.k_proj``, ``self_attn_t.v_proj`` and
  ``self_attn_t.out_proj`` for temporal attention.

Q/K/V names select logical slices of the native PACKED projection. Each
selected slice receives independent A/B matrices. Original packed weights
and biases keep their tensor objects and checkpoint keys. The output
projection and feed-forward layers also retain original weight/bias ownership.
Use exact names, not substring matches or the entire attention module name.

There are ``rank * (in_features + out_features)`` new parameters per selected
projection. For H=200 and feed-forward width 800, rank four adds 8000
parameters per block for the default feed-forward pair. Adding all eight
attention projections adds another 6400 per block. The explicit ten-target
example therefore adds 172800 parameters to a 12-block canonical backbone.

This follows the low-rank update definition in
`Hu et al. (ICLR 2022) <https://github.com/microsoft/LoRA>`_. The archived
attention wrapper mixed batch and sequence axes and could change outputs
with zero updates. This implementation keeps the native batch-first packed
layout and scaled-dot-product attention. It applies dropout to INPUT
features, correcting the archived output-dropout expression.

Insertion, freezing and state
-----------------------------

Zero B preserves deterministic evaluation at insertion. Stochastic training
dropout consumes random numbers, so bitwise equality of stochastic forwards
is not promised. train()/eval() changes dropout behavior without merging
updates into W or modifying base tensor values.

Use ``blocks=[0, 2]`` to choose unique zero-based blocks, or leave blocks as
None to select all. Targets can be added incrementally; duplicate requests
are rejected before modifying any block. Unsupported custom implementations
are rejected rather than silently replacing their behavior. New modules
inherit the base dtype, device and current train/eval mode.

With freeze_base=True, original backbone parameters are frozen and all
installed LoRA/adapter/prompt parameters remain trainable. Task-head flags
are preserved. With False, existing base flags remain unchanged. Parameter
freezing does not force evaluation mode. A task wrapper must use
freeze_encoder=False before installing trainable encoder modules. Frozen
base weights still pass gradients to input signals.

All components are registered modules; deepcopy creates independent owners.
Rebuild backbone/task settings and the SAME tuning configuration before
strict tensor-state restoration with weights_only=True. A native backbone
checkpoint alone does not restore trained low-rank updates or the task head.
Keep source channel order and preprocessing settings with the artifact.

Standalone workflow
-------------------

:doc:`/auto_examples/examples_eegmat_cbramod_lora` trains both feed-forward
and all spatial/temporal Q/K/V/output updates plus the native classifier
head, while keeping all original backbone tensors fixed. It uses subject
isolation, fixed epochs, explicit 19-channel ordering, 100-microvolt units,
recording-level filtering at 256 Hz and four independent 256-to-200 sample
interpolations per window. This differs from the author's direct 200 Hz
pretraining preprocessing and is a teaching protocol, not a benchmark score.

See :doc:`cbramod_heads` for the input contract and complete task state,
:doc:`cbramod_tuning` for post-block adapters, and :doc:`cbramod_prompts` for
input-feature tokens. All three helpers can be combined in any installation
order; rebuild that structure before loading an augmented checkpoint.


Compilation checks
------------------

The standard functional tests do not require a local C++ compiler. To run
the explicit full-graph training checks with a working compiler, use::

    TORCHEEG_TEST_COMPILE=1 python -m unittest test.models.test_cbramod_lora_compile

The three CPU CI runtimes enable this check. It compares eager and compiled
outputs, input and parameter gradients for feed-forward-only, query-only,
full projection and combined adapter/prompt/LoRA configurations, including
native signal masking. No graph-break or eager fallback is allowed.
