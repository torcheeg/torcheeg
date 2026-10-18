CBraMod patch-feature prompts
=============================

:func:`torcheeg.models.add_cbramod_prompts` inserts learned temporal feature
tokens into a CBraMod backbone or task model. Both archived workspace helpers
``get_prefix_model`` and ``get_prompt_model`` performed this same operation.
They map to this one explicit input-token interface. This is not a per-layer
key/value-only prefix mechanism.

Where prompts enter
-------------------

The native patch embedding first computes temporal and spectral features and
adds its positional encoding. Then learned prompts with shape
``[1, electrodes, num_prompts, hid_channels]`` are prepended on the temporal
feature axis, passed through every criss-cross encoder block and cropped from
the result. The original signal patches keep their output order and count.
Both query and key/value positions are added inside the encoder.

The prompts are feature vectors, not EEG samples or reconstructed electrodes.
They do not add physical recording duration or change sampling rate. The
input signal's channel order must stay fixed: each prompt table row belongs
to one configured electrode. A task model supplies its electrode count;
a bare backbone requires an explicit ``num_electrodes`` argument.

Insertion changes attention even before training. Zero-valued prompt tokens
would still change softmax normalization; no zero-initialized identity claim
is made. The helper uses small random normal initialization with standard
deviation 0.02. It does not add positional encoding to the extra tokens,
matching the archived input-token operation.

Freezing and composition
------------------------

Use ``freeze_encoder=False`` on a task wrapper before installation. By default,
``freeze_base=True`` freezes original backbone parameters while keeping learned
prompts and any installed post-block adapters trainable. Task-head flags are
preserved. Setting freeze_base=False preserves existing base flags. Create
the optimizer AFTER installing the tuning modules.

Freezing parameters does not force evaluation mode. Native dropout follows
normal model.train()/model.eval() calls, and new prompt parameters inherit
the backbone dtype/device. Autocast features receive prompts in their current
floating dtype, while gradients return to the parameter's stored dtype.
Input gradients remain available through frozen base parameters.

Post-block adapters and input prompts can be installed in either order;
previously installed tuning parameters remain trainable. Duplicate prompt
installation and incompatible electrode counts are rejected before modifying
the model. Unsupported custom encoders are rejected rather than silently
replacing their forward behavior.

State and independent copies
----------------------------

Prompts use a registered encoder module, with no closure capturing another
model. Native layer and parameter ownership and checkpoint keys are retained.
The extra backbone state key is ``encoder.prompt_tokens``; in a task wrapper
it is ``backbone.encoder.prompt_tokens``. deepcopy creates independent base
and prompt parameters. All native forward/feature interfaces use the same
prompted encoder and return the original signal-patch geometry.

Load a complete local author backbone checkpoint BEFORE prompt installation.
To restore trained prompts, save the backbone/task/tuning constructor settings
and complete tensor state, rebuild that structure, then load it strictly with
weights_only=True. An original backbone checkpoint alone does not restore
learned prompts or a task head. No data or weights are downloaded by the helper.

Standalone workflow
-------------------

:doc:`/auto_examples/examples_eegmat_cbramod_prompts` trains four feature prompts
per electrode and the native multi-layer classifier head on subject-separated
EEGMat data. It records channel order, signal units, filtering, physical time,
independent per-patch interpolation and complete restoration settings. The
small default backbone is random; a complete local author checkpoint is
optional. This is a fixed-epoch teaching protocol, not a benchmark score.
See :doc:`cbramod_heads` for the input contract and :doc:`cbramod_tuning` for
post-block adapters.
