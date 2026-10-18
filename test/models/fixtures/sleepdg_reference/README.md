# SleepDG independent author reference

Source: https://github.com/wjq-learning/SleepDG at
01d209d4fda874933a277a88bccdca6e3835e884, MIT, copyright 2023 Jiquan Wang.
The complete license is retained at LICENSES/SleepDG.txt.

These six models/*.py files retain the author's computational class bodies
verbatim. Only imports from models.* are rewritten as local relative imports
so tests do not alter or depend on a process-global models package.
source-manifest.json records original and adapted SHA256 hashes. No TorchEEG
model computation is used in this reference. Runtime dependency is PyTorch.

The canonical reference has sequence20, channels2, samples3000, width512,
one sequence Transformer, eight heads, MLP512 and five classes. Tests use
explicit dropout0 and float64/eval for deterministic full output and gradient
comparison, and separate tests check training/BatchNorm/dropout behaviour.
Native constructor state and every parameter/buffer path must match.
The reference does not supply pretrained weights or patient data.

The independent alignment reference is `losses/double_alignment.py` from the
same revision. `alignment_reference.py` changes only the domain-index arange
device from unconditional CUDA to `features.device`. Loss formulas, four source
codes and reduction scales are unchanged. Original/adapted hashes and the exact
change are recorded separately in `alignment-manifest.json`. Degenerate cases
and more than four source codes are TorchEEG extensions, not native parity claims.
