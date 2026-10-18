CBraMod parameter-efficient tuning
==================================

Use :func:`torcheeg.models.add_cbramod_adapters` to add trainable residual
bottlenecks to a :class:`torcheeg.models.CBraMod` backbone or one of its
classification, regression or embedding wrappers. Start from an explicit
model configuration and local tensor checkpoint if pretrained weights are
needed. The helper does not load weights, download data or establish a
pretrained performance claim.

Adapter position and initialization
-----------------------------------

One adapter follows each selected COMPLETE criss-cross block: both its
attention residual and feed-forward residual are computed first. This
retains the archived TorchEEG workspace's insertion position. It differs
from adding separate attention and feed-forward adapters in each block.
The general bottleneck adaptation approach is described by
`Houlsby et al. (ICML 2019) <https://proceedings.mlr.press/v97/houlsby19a.html>`_.
The specific post-block placement here is not a claim to reproduce that
paper's full architecture.

For feature width H and bottleneck width A, the adapter computes::

    x + up(dropout(GELU(down(x))))

It has ``H*A + A + A*H + H`` trainable parameters per selected block. The
up-projection weight and bias start at zero, preserving deterministic
backbone outputs at insertion. Hidden dropout uses the configured probability.
Training dropout consumes random numbers, so stochastic forwards are not
promised to be bitwise equal before and after insertion.

Use ``blocks=[0, 2]`` to select unique zero-based blocks, or leave blocks as
None to select all blocks. Repeated adapter insertion into a selected block
is rejected before changing the model. Install adapters before constructing
the optimizer.

Freezing, modes and state
-------------------------

With ``freeze_base=True`` the helper freezes the backbone's original
parameters and keeps all previously installed adapters trainable, including
those in other blocks. Task-head parameter flags remain unchanged. With
``freeze_base=False``, existing base flags remain unchanged. A task wrapper
must use ``freeze_encoder=False`` before insertion: its force-evaluation
behavior would conflict with trainable modules inside the encoder.

Parameter freezing does not force the base into evaluation mode. Dropout
follows normal model.train()/model.eval() calls. Newly added adapters inherit
the selected block's current dtype, device and mode. Frozen parameters still
allow gradients to propagate to input signals. The CBraMod backbone uses
GroupNorm and LayerNorm, not running-statistics BatchNorm.

The helper uses registered modules, with no monkey-patched forward closures.
Base component and parameter objects keep their ownership and state keys.
Adapter keys live under ``encoder.layers.<index>.adapter`` in the backbone.
Normal dtype/device moves, deepcopy and augmented tensor states are supported.
To restore a task, rebuild its backbone and task configurations, reinstall
adapters with the SAME settings and block selection, then load the complete
state strictly from a tensor-only checkpoint. Loading only the original
backbone does not restore learned adapters or the task head.

Real-data example
-----------------

:doc:`/auto_examples/examples_eegmat_cbramod_adapters` trains post-block
adapters and the native multi-layer classifier head while keeping original
backbone tensors fixed. It uses explicit 19-channel ordering, 100-microvolt
units, whole-recording FIR preprocessing at 256 Hz and four independent
one-second 256-to-200 sample interpolations per window. Subjects remain
separate and epochs are fixed. Constructor/tuning/preprocessing settings and
all held-out logits are saved for complete restoration.

The small default backbone is random; a local complete author checkpoint is
optional. This is a teaching/functional protocol, not an EEG-Arena benchmark
or a validated clinical model. See :doc:`cbramod_heads` for the raw input and
physical-time contract. Other original tuning capabilities, including
prompt/prefix and LoRA, are separate from this adapter helper.
