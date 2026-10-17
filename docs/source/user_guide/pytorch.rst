PyTorch compatibility
=====================

TorchEEG models are PyTorch modules. You can use a normal PyTorch training loop
or a TorchEEG trainer. Select a compatible set of Python, PyTorch, NumPy, graph
extensions, and trainer dependencies before installing an experiment.

Verified runtimes
------------------

The shared 1.1.4 development snapshot passed all 555 repository tests in three
separate Linux CPU environments with Python 3.10.20 and NumPy 2.2.6:

.. list-table::
   :header-rows: 1

   * - PyTorch
     - Graph extensions
     - Result
   * - 2.4.1+cpu
     - Matching PyG pt24 wheels
     - 555 tests passed
   * - 2.6.0+cpu
     - Matching PyG pt26 wheels
     - 555 tests passed
   * - 2.14.1+cpu
     - torch-scatter 2.1.2 and torch-sparse 0.6.18 built against this runtime
     - 555 tests passed

These runs also used SciPy 1.15.3, MNE 1.12.1, Lightning 2.6.6, and
TorchMetrics 1.9.0. The existing GitHub test job separately checks Python 3.8
and PyTorch 2.4.1. Package metadata permits additional versions; those ranges
are not a claim that every combination has been tested. Older 1.x runtimes,
CUDA, MPS, distributed training, and dynamic deployment shapes need their own
checks. Optional pretrained image-FID execution is not established by these
CPU runs.

Install graph extensions compiled for your PyTorch, device, and Python version.
Do not reuse extension binaries from a different PyTorch release. See
:doc:`../installation` for the library installation workflow. Keep an existing
training environment intact when testing a new runtime.

Compile a training step
-----------------------

:doc:`../auto_examples/examples_ccnn_compile` uses ``torch.compile`` with the
default Inductor backend and a complete graph. It checks predictions, gradients,
and one optimizer update against eager CCNN execution. The first call includes
compilation work; it is not a steady-state speed measurement. A graph break or
unsupported operation must be investigated for the intended model and inputs.
See the `PyTorch compile reference
<https://docs.pytorch.org/docs/2.14/generated/torch.compile.html>`_.

Export an inference model
--------------------------

:doc:`../auto_examples/examples_eegnet_export` makes a frozen evaluation copy of
EEGNet, exports its static input shape, saves the program, reloads it, and checks
predictions. The original model retains its training mode and gradient flags.
Use the copy when exporting constrained EEGNet weights, including with older
exporters. Export is an inference workflow; it does not export the trainer,
offline preprocessing, or subject partitions. See the `PyTorch export guide
<https://docs.pytorch.org/docs/2.14/export.html>`_.

Use mixed precision
--------------------

:doc:`../auto_examples/examples_ctnet_autocast` runs a CPU bfloat16 CTNet training
step. The model parameters remain float32. Autocast covers the forward pass and
loss; backward runs outside the context. The example checks finite loss,
gradients, and updated parameters. Hardware and operator support determine
speed and numerical behavior. CUDA float16 and its gradient-scaling workflow
require separate checks. See the `PyTorch automatic mixed precision guide
<https://docs.pytorch.org/docs/2.14/amp.html>`_.

Interpret the examples
----------------------

These three examples use synthetic tensors to demonstrate API behavior. They
do not estimate EEG classification accuracy or benchmark performance. They are
standalone scripts with fixed model configurations. Run them directly: the
documentation gallery does not execute training examples during its build.
The scripts report their actual PyTorch version and device.

For actual recordings, start with
:doc:`../auto_examples/examples_eegmat_eegnet` or
:doc:`../auto_examples/examples_eegmat_ctnet`. Verify preprocessing, subject
isolation, numerical behavior, and checkpoint restoration in the target
environment before a larger run.
