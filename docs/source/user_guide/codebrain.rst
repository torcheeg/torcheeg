CodeBrain pretrained encoder
============================

CodeBrain EEGSSM combines temporal and spectral patch embeddings, structured
global convolution, local attention and gated residual blocks. TorchEEG keeps
the complete released EEGSSM state, including both token prediction heads.
The separate :doc:`TFDual tokenizer <codebrain_tokenizer>` has its own complete
interface and checkpoint; it is not part of this encoder checkpoint.

Source and checkpoint
---------------------

Cite Ma et al., `CodeBrain: Bridging Decoupled Tokenizer and Multi-Scale
Architecture for EEG Foundation Model <https://arxiv.org/abs/2506.09110>`_,
ICLR 2026. The `author source <https://github.com/jingyingma01/CodeBrain>`_ at
``22d350caf68246d2fda4f630ef837420db3fb130`` and
`official checkpoint <https://huggingface.co/YjMajy/CodeBrain>`_ use Apache-2.0.
See ``LICENSES/CodeBrain.txt`` for attribution and license notices. Global
convolution follows SGConv by Li et al. (2022).

Use ``CodeBrain.pth`` from revision
``bef08d2fdb1759685371cc635aad21ce59163689``. Its SHA256 is
``d9714b8732c9883a04d022ee66254cd578ae1fa27f5458e6ab7f1aa96e9a7352``.
The complete state contains 269 tensors with a ``module.`` prefix: residual
width 200, eight blocks, ten kernel scales, 200 samples per patch and two
4096-way vocabulary heads. ``CodeBrain_Tokenizer.pth`` is a different model;
do not pass it to this loader.

.. code-block:: python

   import torch
   from torcheeg.models import CodeBrain

   model = CodeBrain.from_pretrained('./CodeBrain.pth').eval()
   eeg = torch.randn(2, 19, 4, 200)  # synthetic shape check
   with torch.no_grad():
       features = model(eeg)  # [2,19,4,200]

Loading is tensor-only on CPU with ``weights_only=True`` and no unsafe
fallback. Every key, shape, dtype and finite value is checked. Kernel norms
must be positive. No download or remote code is executed. The official file
is a flat state dictionary; ``prefix=''`` supports a complete unprefixed
export. A mismatched or missing state raises an error instead of silently
using randomly initialized parameters.

Signals, features and masks
---------------------------

Supply floating ``[batch,electrodes,patches,200]`` tensors, with one-second
nonoverlapping patches at 200 Hz. Channel order matters to the positional
convolution. The model does not infer channel names, align a montage, filter,
scale or normalize signals. Keep the input protocol and electrode order
explicit. Different electrode/patch counts work in the backbone; a downstream
flattening head requires fixed counts. All singleton dimensions are preserved.

``model(eeg, mask)`` accepts a boolean ``[batch,electrodes,patches]`` mask.
Selected raw patches are replaced by the frozen native zero token before
convolution and FFT. The native spectral path uses the magnitude of the
forward-normalized real FFT. Masked input samples therefore receive zero
gradient through this replacement.

``model.predict_tokens(eeg, mask)`` returns time and frequency logits of shape
``[number_of_selected_patches,4096]`` each, in the boolean mask's flattened
order. A mask is required; an empty mask gives empty logits. These are
pretrained vocabulary predictions, not reconstructed EEG or downstream class
labels. Continuous features do not use the two vocabulary heads, but their
complete state remains available for export and token prediction.

The author attention window is one, so only each token's diagonal attention
entry is visible. TorchEEG preserves that computation and places the mask on
the input device, replacing the author's fixed CUDA call. Three-dimensional
channel dropout uses ``Dropout1d``, preserving the author's current
``Dropout2d`` behavior without relying on its announced interpretation change.
The final residual branch is discarded by the native group: TorchEEG retains
its parameters but avoids evaluating that unused branch, preserving unused
gradient behavior during compilation.

Kernel buffers and export
-------------------------

The released checkpoint has uninitialized ``kernel_norm`` buffers. On the
first forward, each global convolution computes its full learned kernel's
norm and caches it with ``kernel_norm_initialized=True``. These norms depend
on weights, not EEG values, batch membership or labels. Subsequent forwards
reuse them; they are not recomputed after every optimizer step. Preserve
these buffers in a trained export to match native behavior.

For compilation, perform an ordinary warmup forward first. TorchEEG tracks
the initialized state without reading a tensor condition in the compiled
forward. Loading a complete state restores that initialization flag. Legacy
public weight normalization retains official ``weight_g``/``weight_v`` names;
its PyTorch deprecation warning remains. Check numerical behavior on the
runtime and backend you plan to use.

Classify and train
------------------

.. code-block:: python

   from torcheeg.models import CodeBrainClassifier
   from torcheeg.trainers import ClassifierTrainer

   classifier = CodeBrainClassifier(model, num_electrodes=19, num_patches=4,
                                    num_classes=2, freeze_backbone=True)
   trainer = ClassifierTrainer(classifier, num_classes=2, accelerator='cpu')
   # trainer.fit(train_loader, val_loader, max_epochs=10)

The new flatten/ELU/dropout head follows the source downstream head layout,
with configurable electrode, patch and class counts. It returns
``[batch,num_classes]`` logits for cross-entropy, including two logits for
binary tasks. The author binary examples instead use a one-logit BCE head;
pretrained encoder support does not imply a pretrained downstream head.
Frozen encoder parameters remain unchanged and encoder dropout stays disabled;
the initial kernel buffers still initialize as described above.

See :doc:`../auto_examples/examples_eegmat_codebrain` for a standalone real-data
workflow. It resamples whole EEGMat records to 200 Hz, filters at 0.3–75 Hz
with a 50 Hz notch, divides microvolts by 100 and makes four one-second
patches per window. The exact EEGMat source electrode order is saved; no
claim is made that it matches the pretraining montage or record trimming.
The last complete subject is held out, with fixed training epochs on all
other subjects. Full state, constructor/provenance JSON, source/weight hashes
and predictions are saved and restored strictly with exact predictions.
This is a functional teaching run, not a published benchmark. Gallery
rendering does not execute its training code.
