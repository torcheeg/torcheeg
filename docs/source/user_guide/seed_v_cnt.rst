SEED-V Neuroscan CNT encoding
=============================

:class:`torcheeg.datasets.SEEDVDataset` reads the provider's raw Neuroscan CNT
files through MNE. The default ``data_format='auto'`` remains unchanged.
The keyword-only option also accepts ``'int16'`` and ``'int32'``. Only choose
an explicit encoding when it has been verified with the data provider;
selecting the wrong width can produce incorrect EEG without an obvious error.

Some files have invalid sample-count fields. With older MNE and NumPy 2, this
can appear as ``Python integer 283467793224 out of bounds for int32`` during
format inference. MNE 1.12 added sample-count recomputation for supported small
files, but upgrading alone does not identify an ambiguous sample encoding.
A compatible reader and a verified explicit encoding may both be necessary.
This does not repair arbitrary corruption, ANT Neuro CNT files, or unsupported
large-file headers. TorchEEG does not guess a format or silently retry a
failed read using a different width. Unrelated reader errors are propagated.

For example, **only if the provider confirms int16**:

.. code-block:: python

   from torcheeg.datasets import SEEDVDataset

   dataset = SEEDVDataset(
       root_path='./EEG_raw',
       data_format='int16',
       io_path='./seed-v-verified-int16-cache',
   )

Before constructing an explicit-format reader, the header sample count is
checked against a conservative bound from the file size, all channels and
the chosen bytes per sample. For a file smaller than 2 GiB, MNE 1.12+ can recompute an impossible count
from its event table when the caller has specified the encoding. Otherwise
TorchEEG rejects it before constructing the reader. A second bound is checked
before ``get_data`` can allocate the corresponding array. This bound is not a complete file
integrity check and cannot determine whether the chosen encoding is correct.

Changing the encoding changes the cache identity. Preserve existing caches
and select a new ``io_path`` when changing reader settings. The channel
selection, MNE voltage calibration, 1000-to-200 Hz downsampling, filtering,
trial annotations and label mappings are not changed by this reader option.

The synthetic CNT tests exercise the full MNE reader with known int16/int32
wire values and physical calibration, the exact unsigned sentinel header,
cache identity and the allocation guard. They are not a claim that the
original issue reporter's unavailable recording has been repaired.
