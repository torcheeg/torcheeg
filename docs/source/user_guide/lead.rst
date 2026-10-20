LEAD with an external backend
=============================

``LEAD`` adapts an explicitly supplied native author model to TorchEEG's
``[batch,electrodes,samples]`` convention. It retains the complete native
model and returns contextual feature vectors. ``LEADClassifier`` adds a new
linear task head and supports either a frozen probe or encoder fine-tuning.

Source boundary and official checkpoint
---------------------------------------

Cite Wang et al., `LEAD: An EEG Foundation Model for Alzheimer's Disease
Detection <https://openreview.net/forum?id=AigNTyxcvH>`_, TMLR, 2026.
The `author source <https://github.com/DL4mHealth/LEAD>`_ at
``c7e21e112a3965b09604b2098e93095880a940dd`` and
`official checkpoint <https://huggingface.co/YiheWang/LEAD>`_ at
``8ac42d5a595a3c080c36a6f84932ac1ba10ee857`` specify CC-BY-NC-SA-4.0.
Obtain and use those materials separately under their own terms.
See ``LICENSES/LEAD-BACKEND-NOTICE.txt`` for attribution and links.

TorchEEG bundles the interface, metadata persistence and task head. It does
not bundle, import, install or download the author implementation or weights.
Backend construction is an explicit caller action using trusted local code.
The complete runnable example below constructs that backend from a pinned
local checkout and rejects tracked changes or a different source revision.

The released P-Base state contains **367 tensors**, 3,317,632 elements:
12 factorized-attention layers, width 128, eight heads, feed-forward width
256, patch length/stride 50, and a learned 1024-position temporal table.
It also stores sampling-rate vectors for 50, 100 and 200 Hz. Its SHA256 is
``3232e93b5a67a7f3141b16b5ae70b09ae94e65163ccff79755119fbfa598c8db``.

The current author pretraining constructor creates a learned contrastive
projection head, but that head is **absent from this released state**.
The example explicitly configures it as ``nn.Identity()`` and enables the
three stored sampling-rate vectors. It then strictly loads every tensor.
It does not leave a random projection head, discard rate embeddings or
claim pretrained contrastive projection weights that are not in the file.
This interface targets features and downstream training; it does not
reproduce the original pretraining schedule.

Explicit construction and safe weight loading
---------------------------------------------

After constructing the documented native backend, wrap it with explicit
ordered channel names and physical metadata:

.. code-block:: python

   from torcheeg.models import LEAD, LEADClassifier

   # backend is the explicitly constructed local native author model.
   features = LEAD.from_pretrained(
       './checkpoint.pth', backend, channel_names=channel_names,
       sampling_rate=200, dataset_id=0
   )
   classifier = LEADClassifier(features, num_classes=2, freeze_backbone=True)

``from_pretrained`` reads a local flat tensor state on CPU with
``weights_only=True``. Keys, shapes, dtypes and finite values must match the
complete supplied backend. There is no unsafe pickle fallback, partial
matching, head removal or embedding resizing. A common key ``prefix`` can
be supplied explicitly. Complete safetensors files are also supported with
the ``foundation`` extra. Loading a checkpoint does not construct or execute
backend source; that separate caller-supplied object already exists.

Inputs and feature semantics
----------------------------

The floating input is ``[batch,electrodes,samples]``. ``channel_names`` must
match the native coordinate-map order, and ``sampling_rate`` is physical Hz.
There is no inferred filtering, reference, amplitude scaling, resampling or
montage. One wrapper represents one input-compatible dataset/channel layout.
The backend's configured rate embeddings must include the requested rate.

The adapter transposes to native time-major layout and calls ``encode``.
Native metadata columns are disease, subject, sampling frequency and dataset
ID. The first two are always **neutral zeros**: neither true labels nor
subject identities enter feature extraction. The remaining columns come
from explicit constructor settings. The released 3D path encodes the supplied
montage coordinates; it does not infer unknown or bipolar electrode geometry.

Native features take the last temporal token per channel, apply GELU and
mode-dependent dropout, then average across channels. Output shape is
``[batch,128]`` for P-Base. The new classifier returns raw task logits.
A frozen backbone stays in eval mode when the classifier trains.
With ``freeze_backbone=False``, ordinary gradients reach the native encoder.
The wrapper never switches to a smaller substitute model.

Geometry and checkpoint restoration
-----------------------------------

The native author checkpoint omits runtime electrode coordinates. Rebuilding
those coordinates with another montage table can change predictions while
all learned weights remain identical. The adapter saves the actual
``electrode_coords``, neutral physical metadata and an ordered-channel
fingerprint alongside all native tensors. A complete P-Base adapter therefore
has 370 state entries; the two-class task head adds two more.

Strictly restoring an adapter requires matching channel names/order,
sampling rate and dataset ID. It restores the saved coordinates instead of
silently replacing them with the current lookup table. The native call binds
that snapshot after a restore or dtype/device conversion. Tests also verify ``to_empty`` followed by strict loading.
Keep channel names, source revision and preprocessing in exported metadata;
trained weights alone do not describe the complete EEG protocol.

Validation and optional CI
--------------------------

The ordinary interface tests need no author source. A separate
``LEAD external backend`` GitHub workflow explicitly obtains the pinned,
separately licensed native source/checkpoint and runs
``test.models.test_lead_native``. The native test checks all 367 tensors,
actual eval/train outputs, input and every parameter gradient, and exact
restoration after changing the runtime coordinate lookup. Run it locally by
setting ``TORCHEEG_LEAD_BACKEND_ROOT`` and ``TORCHEEG_LEAD_CHECKPOINT``.
Without both explicit paths that optional test is skipped; interface and
workflow tests remain active.

On Torch 2.6 CPU, the complete native model and wrapper were also checked
with ``torch.compile(backend='aot_eager', fullgraph=False)``. Outputs and all
gradients agree with eager execution. The source's tensor-valued metadata
lookups require ordinary graph breaks; fullgraph execution and optimized
GPU throughput are not established by that check.

Independent real EEGMat workflow
--------------------------------

See :doc:`../auto_examples/examples_eegmat_lead` for the complete local-backend
construction, strict loading, training, export and restoration workflow.
Clone the author repository separately and select the pinned revision:

.. code-block:: console

   git clone https://github.com/DL4mHealth/LEAD.git ./LEAD
   git -C ./LEAD checkout c7e21e112a3965b09604b2098e93095880a940dd
   python examples/examples_eegmat_lead.py --root-path ./eegmat \
       --backend-root ./LEAD --checkpoint ./checkpoint.pth --epochs 1

Supply the complete separately obtained P-Base checkpoint as ``checkpoint.pth``.
The example resamples whole EDF records from 500 to 200 Hz, then applies
0.5–45 Hz FIR/50-Hz notch filtering. It standardizes each four-second
window/channel using population mean/std with a 1e-6 floor and keeps all 19
source electrodes. The named 1020 montage provides explicit geometry;
the actual coordinates and lookup version are saved. No population statistics
are fitted using held-out subjects.

The last complete subject is held out; only a new head trains on the others
for fixed epochs. Full native/adapter/head state, coordinates, ordered names,
physical metadata, source/checkpoint hashes and predictions are exported and
strictly restored. This small teaching transfer run is not an AD benchmark,
a clinical validation or a published-score reproduction. Gallery rendering
does not run the training script or execute the supplied backend.
