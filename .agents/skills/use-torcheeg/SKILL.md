---
name: use-torcheeg
description: Use TorchEEG to load EEG data, apply transforms, split samples, and train or evaluate a PyTorch model. Use for workflows built with this library.
---

# Use TorchEEG

Build a runnable EEG workflow with the installed TorchEEG version.
This skill covers library usage. It does not operate a headset or a local EEG service.

## Select the workflow

Check the installed version and the public API before selecting a component.
For a source checkout, inspect `torcheeg/`, `examples/`, and `docs/source/`.
For an installed release, use its matching documentation.
Do not assume that an unmerged feature is in a published package.

Use [the API reference](https://torcheeg.readthedocs.io/en/latest/torcheeg.datasets.html)
and [the examples](https://torcheeg.readthedocs.io/en/latest/auto_examples/index.html).
Select an example with the same signal format, task, and model family.

Resolve these inputs from the task and data:

- Dataset format and file location.
- Sampling rate, units, channel names, and channel order.
- Window length and overlap.
- Label field, class mapping, and number of output classes.
- Evaluation groups, such as subjects, sessions, or trials.

Request only information that remains necessary and unavailable.

## Prepare data and labels

Use the dataset's documented constructor and expected file layout.
Do not substitute a different dataset because its file format looks similar.
Use `offline_transform` for preprocessing saved in the dataset cache.
Use `online_transform` for operations performed when a sample is read.
Use `label_transform` to select and map metadata into the training target.

Check one sample and its metadata before training.
Confirm the tensor shape, finite values, units, and label range.
For classification, use integer targets from zero through `num_classes - 1`.
For regression, use the target format required by the selected trainer and loss.
Keep electrode order consistent with the model and any grid or graph constants.

Use a new cache path when source data or offline preprocessing changes.
Check the split cache when sample metadata or evaluation settings change.
Existing caches can otherwise reuse an earlier experiment's data or partitions.
Preserve existing caches unless their removal is authorized.

## Split and train

Choose a splitter that matches the evaluation protocol.
For unseen-subject evaluation, use a subject-based splitter.
For unseen-trial evaluation, keep all windows from one trial in the same group.
Check actual group IDs in every partition before training.
Overlapping windows from one recording can leak information across random sample splits.

For standard holdout evaluation, fit learned preprocessing on training data only.
Select model settings with training and validation data.
Reserve test labels for the final evaluation.
If a protocol uses unlabeled target data during training, state that use explicitly.
Do not treat that protocol as evaluation without access to target data.

Match the model's documented input shape.
EEGNet takes `[batch, 1, electrodes, time]`; other model families can need different inputs.
Graph models also need their documented graph representation.
Models can return logits, tuples, or other outputs; check the selected class.
Pass logits directly to cross-entropy loss when that model returns logits.
Use `ClassifierTrainer`, `RegressorTrainer`, or a plain PyTorch loop as the task requires.

This synthetic EEGNet example checks one optimization step. It measures no EEG accuracy.

```python
import torch
from torcheeg.models import EEGNet

model = EEGNet(num_electrodes=32, chunk_size=128, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
eeg = torch.randn(4, 1, 32, 128)
labels = torch.tensor([0, 1, 0, 1])
optimizer.zero_grad()
logits = model(eeg)
loss = torch.nn.functional.cross_entropy(logits, labels)
loss.backward()
optimizer.step()
```

## Verify and report

Check output shape, finite loss, gradients, and a checkpoint round trip.
For evaluation, use `model.eval()` and disable gradient tracking.
Run the complete example on the requested data when those data are available.
Report a missing data source or access requirement without inventing a result.

Record package versions, preprocessing, group splits, seeds, and metric definitions.
Distinguish synthetic checks, real-data runs, and published reference scores.
Link a reported result to its example and saved evidence.
