NeuroRVQ with an external backend
================================

``NeuroRVQ`` wraps a complete explicitly supplied author backend.
``NeuroRVQClassifier`` adds a new task head for either frozen features or
encoder fine-tuning. The adapter never downloads or imports author source.

Source and complete checkpoint
------------------------------

Cite Barmpas et al., `NeuroRVQ <https://arxiv.org/abs/2510.13068>`_.
The `author repository <https://github.com/KonstantinosBarmpas/NeuroRVQ>`_
and `official weights <https://huggingface.co/ntinosbarmpas/NeuroRVQ>`_
use **CC-BY-NC-4.0**. Obtain and use these materials separately under their
terms. TorchEEG distributes its interface, metadata persistence and new task
head; it bundles neither the author source nor the checkpoint.
See ``LICENSES/NEURORVQ-BACKEND-NOTICE.txt``.

Use author source revision ``27c3c134a53cc29c0d1696ebfe65f3e17ea9ab0c``.
The later inference-only implementation removes the prediction heads;
its permissive loader silently ignores released pretraining state. This
integration instead uses the historical complete pretraining implementation.
The EEG-v1 checkpoint at Hub revision
``d944b87f44ae0ba2923b2f10d0518f23f6803b76`` is
``pretrained_models/foundation_models/NeuroRVQ_EEG_foundation_model_v1.pt``.
Its SHA256 is
``d14ef4eae2d4594d5f1f709226a3bb5f362afd84b68050d5da0fa0586aea886a``.
It contains **330 state tensors**, 58,560,632 elements: a four-branch temporal
convolution frontend, twelve shared Transformer blocks of width 200 with ten
heads, 256 temporal positions, 105 spatial rows, and **32 independent
8192-class RVQ prediction heads**. The checkpoint size includes those heads;
the feature encoder alone is much smaller. No heads are discarded.

The example constructs the complete author model explicitly. Parameter-free
``Identity`` views expose its raw feature branches without adding random
normalization weights. All released tensors load strictly and remain in the
exported state. This is an external-backend integration, not a newly bundled
native implementation or a smaller replacement network.

Input and feature semantics
---------------------------

Input is floating ``[batch,electrodes,samples]`` at **200 Hz** with complete
200-sample patches and at most 256 patches per electrode. The caller must
provide ordered electrode names and the exact author electrode vocabulary.
Names are matched ignoring case; unknown or duplicate names are rejected.
There is no automatic channel removal, interpolation, padding or truncation.
The adapter does not infer amplitude scaling, filtering or referencing.

Temporal slots are right-aligned to the end of the 256-row table. Spatial
IDs follow the original **zero-based** vocabulary, including its use of
row zero for the first named electrode as well as the legacy class token.
The integration does not shift those IDs. Electrode/time flattening is
channel-major. The four raw contextual branches concatenate to width 800.
``forward`` averages all patch features to ``[batch,800]``;
``forward(x, return_tokens=True)`` returns ``[batch,electrodes,patches,800]``.
A new classifier produces raw logits. A frozen backbone stays in evaluation
mode while the classifier trains; ordinary fine-tuning permits gradients.

Complete tensor loading and restoration
---------------------------------------

After explicitly constructing the documented backend:

.. code-block:: python

   from torcheeg.models import NeuroRVQ, NeuroRVQClassifier

   backbone = NeuroRVQ.from_pretrained(
       './NeuroRVQ_EEG_foundation_model_v1.pt', backend,
       channel_names=channel_names, channel_vocab=original_vocabulary
   )
   model = NeuroRVQClassifier(backbone, num_classes=2, freeze_backbone=True)

The loader reads local flat tensors on CPU with ``weights_only=True``.
Every key, shape, dtype and finite value must match. Complete safetensors
and an explicit common key prefix are supported. There is no unsafe pickle
fallback, partial match or head removal. Loading weights never constructs
or executes backend code; explicit trusted backend construction is separate.

The adapter also saves spatial IDs and a fingerprint of ordered names,
original vocabulary and physical sampling rate. Strict adapter restoration
rejects a different order or vocabulary. A complete feature adapter has 332
state entries and a new two-class head adds two. Save these metadata and the
preprocessing protocol with the full state. ``to_empty`` followed by strict
restoration is covered by interface tests.

Symmetric RVQ prediction
-------------------------

``forward_masked(x, mask)`` accepts a same-device boolean mask of shape
``[batch,electrodes,patches]``. Both partitions must contain at least one
patch. It calls the original complete symmetric mask path, returning two
logit tensors shaped ``[32,masked_count,8192]`` and
``[32,unmasked_count,8192]``. Counts include all examples and electrodes;
token ordering is the native boolean-index order. All 32 heads participate
in gradients. Obtain matching target IDs from the separately supplied EEG-v1
tokenizer; the adapter does not invent targets. The tokenizer, full RVQ
training schedule and a real-data pretraining example remain separate work.

Independent real EEG example and validation
-------------------------------------------

See :doc:`../auto_examples/examples_eegmat_neurorvq` for complete explicit
construction, strict weight loading, a frozen linear probe, full export and
exact prediction restoration. Obtain the author repository separately:

.. code-block:: console

   git clone https://github.com/KonstantinosBarmpas/NeuroRVQ.git ./NeuroRVQ
   git -C ./NeuroRVQ checkout 27c3c134a53cc29c0d1696ebfe65f3e17ea9ab0c
   python examples/examples_eegmat_neurorvq.py --root-path ./eegmat \
       --backend-root ./NeuroRVQ --checkpoint ./NeuroRVQ_EEG_foundation_model_v1.pt --epochs 1

The example rejects a different author revision or tracked source changes.
It preserves all 19 EEGMat EDF electrodes and source reference, resamples
whole records from 500 to 200 Hz, applies MNE FIR 0.5–45 Hz and 50-Hz notch,
and standardizes each four-second window/channel with population mean/std
and a 1e-6 floor. This is an explicit teaching transfer normalization, not a
reproduction of the original benchmark preprocessing. No population
statistics are learned from the held-out subject. The last complete subject
is held out and only a new head trains on the others for fixed epochs.
Gallery rendering does not execute the training workflow or author backend.

Independent interface tests require no author code. Optional native tests
require explicit ``TORCHEEG_NEURORVQ_BACKEND_ROOT`` and
``TORCHEEG_NEURORVQ_CHECKPOINT`` paths. They compare complete native eval/train
features, symmetric logits, EEG gradients and every parameter gradient,
and verify sample isolation. Without these paths only that external test
is skipped. A separate GitHub workflow obtains the pinned author materials
under their own terms and runs the same complete oracle.
