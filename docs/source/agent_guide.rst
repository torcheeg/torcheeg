Using TorchEEG with an assistant
================================

TorchEEG provides two short guides for coding assistants and other tools.
Use the guide that matches the task:

* ``AGENTS.md`` describes repository changes, tests, documentation, and contribution conventions.
* ``.agents/skills/use-torcheeg/SKILL.md`` describes how to use the library in an EEG workflow.

The library skill covers data formats, preprocessing, cache reuse, group splits,
model inputs, training, and verification. It does not need a running EEG device
or an application service. Its synthetic example checks software behavior.
It does not measure classification accuracy on EEG data.

Use the skill
-------------

In a source checkout, read the skill file directly or use a tool that supports
project skills in ``.agents/skills``. Tool support varies. If automatic discovery
is unavailable, give the file to the assistant as task guidance.

For a separate analysis project, copy the ``use-torcheeg`` folder into that
project's skill directory. Keep the guide aligned with the installed library
version. These files do not install Python dependencies or grant access to data.

* :download:`Download the library skill <../../.agents/skills/use-torcheeg/SKILL.md>`
* :download:`Download the contributor guide <../../AGENTS.md>`

Browse the API and examples
---------------------------

Start with :doc:`installation`, then select a dataset and model from
:doc:`torcheeg.datasets` and :doc:`torcheeg.models`.
Use :doc:`torcheeg.transforms` and :doc:`torcheeg.model_selection` to make the
input format and evaluation protocol explicit.
See :doc:`auto_examples/index` for complete workflows.

A plain-text documentation index is available as `llms.txt <llms.txt>`_.
The index helps tools find the relevant pages. It does not guarantee discovery
by a search engine or an assistant. Confirm the package version before using
an API from the latest documentation.
