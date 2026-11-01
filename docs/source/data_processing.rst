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

    import torch
    from torcheeg import transforms

    def preprocess_sample(sample):
        remove = transforms.BaselineRemoval(baseline=sample['baseline'])
        return {**sample, 'eeg': remove(sample['eeg'])}

    def prepare_training(sample):
        return {
            'eeg': transforms.ToTensor()(sample['eeg']),
            'labels': torch.tensor(int(sample['info']['valence'] > 5)),
        }

    train = dataset.with_transform(preprocess_sample).with_transform(prepare_training)
    equivalent = dataset.with_transform(transforms.Compose([
        preprocess_sample,
        prepare_training,
    ]))

Both forms execute the functions in order each time a sample is read. The source
dataset and stored signals remain unchanged. Compose flattens nested pipelines.
The training view can be passed to a PyTorch DataLoader; TorchEEG trainers consume
batches with ``eeg`` and ``labels`` fields.

Store transformed samples
-------------------------

A transform that produces storage samples retains ``eeg``, ``info`` and an
optional ``baseline``. It can be materialized into a new dataset:

.. code-block:: python

    from torcheeg.datasets import MemoryDataset

    processed = MemoryDataset()
    processed.extend(subject.with_transform(preprocess_sample))

Dataset shortcuts
-----------------

Dataset-specific constructors compose their Data class, runner and cache. Their
``transform`` runs once during preprocessing; use ``with_transform`` for work
performed on each read.

.. code-block:: python

    from torcheeg.datasets import DEAPDataset

    dataset = DEAPDataset(
        root_path='./data_preprocessed_python',
        chunk_size=128,
        transform=preprocess_sample,
        io_path='./cache/deap-preprocessed',
        num_worker=4,
    )
    train = dataset.with_transform(prepare_training)

Shortcuts validate their source cache using a manifest and publish completed
builds from a temporary directory. Explicit ``append`` and ``extend`` are
incremental writes: samples written before an iterator fails remain available.

Custom Data
-----------

Implement ``set_records`` to discover records, ``read_record`` to read one record,
``fake_record`` to provide a synthetic payload, and ``process_record`` to yield
sample dictionaries. Override the relevant step to change record preprocessing.
A Data ``transform`` runs on each yielded sample before it leaves the runner.
