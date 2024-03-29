Hooks for EEG datasets
===================

.. contents:: We provide hooks for preprocessing trial signals, session signals, and subject signals. They can leverage global information from trials, sessions, and subjects, and thus can improve the performance of prediction models, especially generalization. Please refer to the paper you are comparing to determine whether you should use it to conduct a fair comparison.
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: torcheeg.datasets

.. autofunction:: before_hook_normalize
.. autofunction:: after_hook_normalize
.. autofunction:: after_hook_running_norm
.. autofunction:: after_hook_linear_dynamical_system