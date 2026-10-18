LaBraM parameter-efficient tuning
==================================

:func:`torcheeg.models.add_labram_adapters` installs registered residual
bottlenecks in :class:`torcheeg.models.LaBraMEncoder` or its downstream
classification, regression and embedding wrappers. See :doc:`labram_encoder`
for complete native state and :doc:`labram_tasks` for sampling, token
normalization and task-head contracts. The existing LaBraM classifier API
remains separate; the helper accepts the encoder and wrappers named here.

Two insertion positions
------------------------

Each selected block receives two independent adapters. The first follows
the complete attention residual, including native gamma_1 scaling and
stochastic depth. The second follows the complete MLP residual, including
gamma_2 scaling and stochastic depth. For either residual result h::

    h + up(dropout(GELU(down(h))))

The learned base scales remain the original Parameter objects at their
original checkpoint paths. Blocks with layer_scale=0, which omit the scales,
are supported. Native component objects, parameter identities and state keys
are retained. New adapters inherit the selected block's device, dtype and
train/eval mode; existing mixed child-module modes remain intact.

The down projection uses small random weights. Up weights and biases start
at zero, so insertion preserves deterministic base outputs. The archived
workspace used small nonzero up weights; zero initialization follows the
existing EEGPT/CBraMod adapter convention and gives an explicit identity
start. A first optimization step can update the up projection; subsequent
steps also train the down projection. Training dropout consumes random
numbers and can change subsequent stochastic-depth masks, so stochastic
training forwards are not promised bitwise identity.

Selection and trainability
---------------------------

Pass ``blocks=[0, 3]`` for unique zero-based block indices or None for all
blocks. Invalid, repeated, already-adapted or unsupported custom selections
raise an error before model mutation. Disjoint installations are supported.
Each block owns separate attention and MLP adapters; weights are not shared
between blocks.

Default ``freeze_base=True`` freezes every native encoder parameter and
keeps installed adapters trainable, including adapters from prior disjoint
installations. It preserves task-head flags. ``freeze_base=False`` preserves
existing base/adapter trainability flags. This is a parameter policy, not
a request to change module training modes.

Task wrappers must use ``freeze_encoder=False`` when installing trainable
encoder modules. Their stronger freeze_encoder=True mode would keep the
whole encoder, including adapter dropout, in evaluation mode. For example::

    backbone = LaBraMEncoder()
    classifier = LaBraMClassifier(backbone, ['Fp1', 'C3', 'C4'], chunk_size=800,
        sampling_rate=200, freeze_encoder=False)
    add_labram_adapters(classifier, adapter_size=16, dropout=.1)
    optimizer = torch.optim.Adam(
        [p for p in classifier.parameters() if p.requires_grad], lr=.001)

The default flattening task head can contain more trainable parameters than
the adapters. Encoder-parameter freezing alone does not make the whole task
head compact. Set pooling='mean' and head_channels=() for a linear mean-pooled
head when that is the intended task architecture; record and validate that
choice separately from the default flattening protocol.

Create optimizers after installing modules. Input gradients remain available
through frozen native weights. An optimizer receives new adapter parameters
only after they exist in the module tree. Dropout follows normal train/eval
calls. If the base was configured with nonzero dropout or stochastic depth,
those modes still apply while its parameters are frozen.

Loading and restoration
------------------------

Load a complete local native encoder before installing adapters. The encoder
factory does not silently accept an augmented checkpoint as native state.
For a trained augmented task, rebuild the encoder and wrapper from their
saved configurations, install adapters with the same configuration and
selected blocks, then strictly load complete tensor state. Do not reload only
the encoder and call the task restored.

Registered modules support copying, device/dtype moves, state dictionaries,
fixed-window full-graph compilation and strict export. For Fourier pipelines,
use ``torch.compile(model, fullgraph=True, dynamic=False)`` as described in
the task guide; symbolic FFT inputs are not claimed. No checkpoint download,
unsafe pickle fallback or automatic optimizer-state restoration occurs.

Teaching workflow
------------------

:doc:`/auto_examples/examples_eegmat_labram_adapters` trains adapters and a
new classification head on public EEGMat background/arithmetic labels,
holding out a complete subject. It uses whole-record 200 Hz resampling,
.1--75 Hz bandpass and 50 Hz notch MNE FIR, source channel order/reference,
microvolts divided by 100 and four-second windows. Native encoder dropout
and stochastic depth are zero in this example; adapter/head dropout is .1.

The default encoder is small and random. A verified local canonical base
checkpoint is optional. The example fixes epochs, fits no cross-subject
statistics and does not select settings on test labels. It saves complete
native/adapter/head state, all three constructor configurations,
preprocessing/channel/partition settings, trainable parameter count and every
held-out output/target. This verifies functionality and restoration, not a
published adapter or EEG-Arena score.


Attention prefixes
------------------

:func:`torcheeg.models.add_labram_prefix` prepends learned keys and values to
selected attention layers. Each layer owns independent prefix_tokens with
shape [prefix_length, 2, heads, head_width], initialized from a normal
distribution with standard deviation .02. Native queries and keys undergo
optional native per-head normalization before insertion. Learned prefix
keys bypass that normalization, preserving the archived workspace's order.

Prefixes do not receive channel/time positions or produce queries. The
attention probability matrix has N query rows and prefix_length + N key
columns; the output still has N tokens, including the original CLS token.
Even zero prefixes change the softmax denominator, so prefix insertion is
not an identity transformation. The helper does not claim an unchanged base
output or pretrained prefix weights.

Use the same explicit block selection, freezing and restoration rules as
for adapters. Native modules and parameter objects retain their paths.
Prefix state is registered under blocks.<index>.attn.prefix_tokens; this
replaces the archived closure's block-level prefix storage with ordinary
module ownership. Device/dtype moves, copying, strict state restoration,
fixed-window compilation and export include this state. Attention dropout
follows normal train/eval calls; there is no extra prefix dropout.

For example::

    backbone = LaBraMEncoder()
    classifier = LaBraMClassifier(backbone, ['Fp1', 'C3', 'C4'], chunk_size=800,
        sampling_rate=200, pooling='mean', head_channels=(), freeze_encoder=False)
    add_labram_prefix(classifier, prefix_length=10, blocks=[0, 3])
    optimizer = torch.optim.Adam(
        [p for p in classifier.parameters() if p.requires_grad], lr=.001)

Adapters and prefixes can be installed in either order. Both helpers keep
all installed tuning parameters trainable when freeze_base=True, including
prior disjoint installations. Native encoder parameters are frozen and
head flags are preserved. freeze_base=False preserves existing flags and
makes new prefix parameters trainable. Duplicate/custom attention selection
or invalid arguments raise before any selected attention is replaced.
Install all modules before constructing the optimizer. Reconstruct the same
adapter/prefix selections and configurations before strict restoration;
loading native-only weights does not restore a trained augmented task.

:doc:`/auto_examples/examples_eegmat_labram_prefixes` supplies a standalone
subject-held-out workflow with the same explicit native preprocessing as
the adapter example. It trains prefixes and the default flattened task
head, whose parameter count can dominate the prefixes. It saves complete
native/prefix/head state, all three constructor configurations, preprocessing,
partition settings and every held-out logit/target. A local verified base
checkpoint is optional; the default small encoder is random. This is a
teaching and restoration check, not a published score.


For float16 training, prefer autocast with float32 stored parameters and
GradScaler. Calling model.half() also stores parameter gradients in float16:
small prefix-key gradients can underflow, and small updates can round away.
Loss scale and optimizer epsilon therefore matter even when outputs are
finite. Verify actual updates to both prefix keys and values. The teaching
example uses float32 and does not claim that every pure-float16 objective
will train successfully without appropriate scaling.
