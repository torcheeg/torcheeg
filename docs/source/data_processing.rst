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

Offline and online transforms
-----------------------------

Transforms are callables with one typed input and one typed output. Constructor
arguments supply configuration or auxiliary data, such as a baseline. Sample
functions choose which fields to transform.

.. code-block:: python

    from torcheeg import transforms
    from torcheeg.datasets import DEAPDataset
    from torcheeg.datasets.utils import get_locations

    def preprocess_fn(sample: dict) -> dict:
        eeg_transform = transforms.Compose([
            transforms.BandDifferentialEntropy(),
            transforms.ToGrid(
                get_locations(DEAPDataset.channels, DEAPDataset.grid)),
        ])
        return {
            **sample,
            "eeg": eeg_transform(sample["eeg"]),
            "baseline": eeg_transform(sample["baseline"]),
        }

    def transform_fn(sample: dict) -> dict:
        eeg_transform = transforms.Compose([
            transforms.BaselineRemoval(baseline=sample["baseline"]),
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

Data's ``transform`` runs on each generated sample before storage.
IODataset's ``transform`` runs on each sample read from storage.

.. code-block:: python

    def main():
        data = DEAPData(
            root_path='./data_preprocessed_python',
            transform=preprocess_fn,
        )
        dataset = IODataset(
            eeg_io=LMDBEEGIO('./cache/deap/eeg'),
            info_io=CSVInfoIO('./cache/deap/info.csv'),
            transform=transform_fn,
        )
        dataset.extend(MultiprocessRunner(num_worker=4)(data))
        return dataset

    if __name__ == '__main__':
        dataset = main()

The runner processes and stores samples incrementally. Data construction does
not start preprocessing; the runner consumes it. ``append`` and ``extend`` write
samples without applying the Dataset's online transform. Stored samples contain
``eeg``, ``info`` and optionally ``baseline``. The online function returns the
training fields; DataLoader preserves their names in batches.

To combine sample functions, pass ``transforms.Compose([f, g])`` as the transform.
The output of ``f`` becomes the input of ``g``. Selections preserve the transform.

Dataset shortcuts
-----------------

Dataset-specific constructors compose their Data class, runner and cache:

.. code-block:: python

    dataset = DEAPDataset(
        root_path='./data_preprocessed_python',
        io_path='./cache/deap',
        offline_transform=preprocess_fn,
        online_transform=transform_fn,
        num_worker=4,
    )

``offline_transform`` runs before storage when creating the cache.
``online_transform`` runs on every read. Both parameters accept a callable or
a ``Compose`` pipeline. Label processing stays inside the user's function.
Changing the online function reuses the same cache. Changing the offline
function requires a new cache path. ``num_worker`` controls preprocessing;
PyTorch DataLoader's ``num_workers`` controls parallel training reads.

Shortcuts publish completed caches from a temporary directory. Explicit
``append`` and ``extend`` store samples incrementally.

Custom Data
-----------

Implement ``set_records`` to discover records, ``read_record`` to read one record,
``fake_record`` to provide a synthetic payload, and ``process_record`` to yield
sample dictionaries. Override the relevant step to change record preprocessing.
Pass ``transform`` to the Data constructor to transform yielded samples before storage.
