# TorchEEG contributor guide

TorchEEG is a Python library for EEG analysis with PyTorch.
For library usage, read [.agents/skills/use-torcheeg/SKILL.md](.agents/skills/use-torcheeg/SKILL.md).

## Find the relevant code

- Library code is in `torcheeg/`.
- Functional tests are in `test/`.
- Sphinx sources are in `docs/source/`.
- Standalone workflows are in `examples/`.
- Dependency groups and supported Python metadata are in `setup.py`.
- CI commands and runtime versions are in `.github/workflows/test.yml`.

Read the neighboring implementation and tests before a change.
Use their parameter names, docstrings, comments, and public import structure.
Keep useful contributor attribution and required license notices.
Preserve existing public behavior unless the task requires a documented change.

## Check EEG behavior

State tensor shapes, sampling rates, signal units, channel order, and label mappings.
Keep subjects or trials separate when the evaluation protocol requires this.
For a standard holdout protocol, fit data-dependent preprocessing on training data only.
State any protocol that also uses unlabeled target data.
Keep the test set out of model and parameter selection.
Check cached data when input files or preprocessing settings change.
Do not delete an existing cache to make a test pass.

Test numerical behavior, gradients, metadata, or sample isolation as appropriate.
Shape checks alone cannot establish model or dataset correctness.
Keep synthetic smoke results separate from real-data results and published scores.

## Validate the change

Use an isolated Python environment and the dependency groups needed for the task.
For a source checkout, install with `python -m pip install -e .`.
Do not replace a user's training environment without authorization.

Run the relevant test module first. Then run the checks required by the change.
The complete test command, from the repository root, is:

```sh
python -m unittest discover -s test -t .
```

Build documentation with:

```sh
python -m sphinx -b html -W --keep-going docs/source build/docs
```

The current Sphinx Gallery configuration does not run the training examples.
Run a changed example separately and record its data requirements and result.
Local documentation success does not establish Read the Docs success.
Report unavailable data, missing dependencies, and failed checks explicitly.

## Submit focused changes

Inspect the working tree before editing. Preserve unrelated local changes.
Use separate feature, fix, test, or documentation branches for independent work.
Follow recent commit-message style. Preserve contributor and PR references.
Use the maintainer's existing authorization for pushes, merges, and publication.
Describe the behavior change, validation, and known limits in the PR.
