Data, runners and storage
=========================

Process and store samples
-------------------------

A Data class discovers records, reads each record and yields sample dictionaries.
A record can produce many samples. Runners schedule this work and yield results
incrementally. Dataset storage consumes the iterator without collecting it first.

.. code-block:: python

    from torcheeg.datasets import DEAPData, IODataset
    from torcheeg.io import CSVInfoIO, LMDBEEGIO
    from torcheeg.runners import MultiprocessRunner

    def main():
        data = DEAPData(root_path='./data_preprocessed_python', chunk_size=128)
        dataset = IODataset(
            eeg_io=LMDBEEGIO('./cache/deap/eeg'),
            info_io=CSVInfoIO('./cache/deap/info.csv'),
        )
        dataset.extend(MultiprocessRunner(num_worker=4)(data))
        return dataset

    if __name__ == '__main__':
        dataset = main()

The explicit equivalent of ``extend`` is:

.. code-block:: python

    for sample in runner(data):
        dataset.append(sample)

``SequentialRunner()`` runs in the calling process. ``MultiprocessRunner`` uses
worker processes and ``JoblibRunner`` uses Joblib threads. Parallel runners
preserve record order and bound the queue of pending samples.

Read and select
---------------

.. code-block:: python

    sample = dataset[0]
    eeg = sample['eeg']
    metadata = sample['info']
    baseline = sample.get('baseline')

    first = dataset[:10]
    chosen = dataset[[0, 3, 5]]
    mask = dataset.info['subject_id'].eq('s01.dat').to_numpy()
    subject = dataset[mask]

Selections share signal storage and preserve metadata order. Integer indexing
and iteration return sample dictionaries. Storage keys are kept in the metadata
table; sample ``info`` contains the source metadata.

Transform on reading
--------------------

Array transforms take an array and return a transformed value. Sample functions
choose the fields to transform. Baseline removal binds the sample's baseline
when constructing the array transform.

.. code-block:: python

    from torcheeg import transforms
    from torcheeg.datasets import DEAPDataset
    from torcheeg.datasets.utils import get_locations

    def transform_fn(sample):
        eeg_transform = transforms.Compose([
            transforms.BandDifferentialEntropy(),
            transforms.BaselineRemoval(
                baseline=transforms.BandDifferentialEntropy()(sample["baseline"]),
            ),
            transforms.ToGrid(get_locations(DEAPDataset.channels, DEAPDataset.grid)),
            transforms.ToTensor(),
        ])
        label_transform = transforms.Compose([
            transforms.Select("valence"),
            transforms.Binary(5.0),
        ])
        return {
            "eeg": eeg_transform(sample["eeg"]),
            "label": label_transform(sample["info"]),
        }

    train = dataset.with_transform(transform_fn)

EEG and label transforms each form a Compose pipeline. The function returns
``eeg`` and ``label`` fields for training. DataLoader preserves these names in
batches. Each read performs the transformations; stored signals remain unchanged.

Additional sample functions can be appended with
``dataset.with_transform(f).with_transform(g)`` or
``dataset.with_transform(Compose([f, g]))``. Both execute in the same order.

Dataset shortcuts
-----------------

Dataset-specific constructors compose their Data class, runner and cache.
Use the same sample function when reading from a shortcut:

.. code-block:: python

    dataset = DEAPDataset(
        root_path='./data_preprocessed_python',
        chunk_size=128,
        io_path='./cache/deap',
        num_worker=4,
    ).with_transform(transform_fn)

The constructor's ``transform`` argument is for persistent preprocessing and
must return storage fields: ``eeg``, ``info`` and optionally ``baseline``.
The training function above returns ``label`` instead of ``info`` and therefore
belongs in ``with_transform``. ``append`` and ``extend`` accept storage samples.

Shortcuts validate their source cache using a manifest and publish completed
builds from a temporary directory. Explicit ``append`` and ``extend`` are
incremental writes: samples written before an iterator fails remain available.

Custom Data
-----------

Implement ``set_records`` to discover records, ``read_record`` to read one record,
``fake_record`` to provide a synthetic payload, and ``process_record`` to yield
sample dictionaries. Override the relevant step to change record preprocessing.
A Data ``transform`` runs on each yielded sample before it leaves the runner.
