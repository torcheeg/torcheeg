"""
Train SleepDG across five sleep datasets
=============================================

Use Sleep-EDF, HMC, ISRUC, SHHS and P2018 with one entire dataset held out.
Obtain each source through its provider; this script does not download data.
Cite Wang et al., AAAI 2024, https://arxiv.org/abs/2401.05363, and the dataset
providers. Install the source with ``pip install -e ".[dataset,trainer]"``.

Run the native 20-epoch, two-channel, width-512 architecture::

    python examples/examples_sleepdg_cross_dataset.py \
        --sleepedf-root ./sleep-edf --hmc-root ./HMC/recordings \
        --isruc-root ./ISRUC --shhs-root ./SHHS --p2018-root ./P2018/training \
        --target shhs --output-dir ./sleepdg-run --accelerator gpu

Each source dataset needs at least two participants. Source validation holds
out participants before sequence construction. Target records never enter
training or checkpoint selection. Per-record normalization uses unlabeled
statistics from each retained record, including target records.

This corrected TorchEEG workflow differs from the author's prototype:
source validation is participant-disjoint, batches balance source datasets,
all records are processed, unscored intervals remain gaps, and normalization
uses complete retained runs. Sleep-EDF keeps up to 60 wake epochs on each
side of scored sleep. See the workflow guide for the exact per-source
filters, channel names, references, normalization and corrected boundaries.
It is not a claim of reproducing published accuracy.

The default is 50 training epochs. For a functional check on explicitly
selected source directories, use ``--epochs 1 --batch-size 4 --train-batches 2
--val-batches 2 --test-batches 2``. Batch limits are recorded in the report.
Use separate, new caches after changing data or preprocessing. Resume only
an owned, trusted end-of-epoch Lightning checkpoint with ``--resume PATH``;
keep source partitions and the planned scheduler horizon unchanged.
"""

import argparse
from copy import copy
import json
import math
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from torcheeg import transforms
from torcheeg.datasets import (HMCDataset, ISRUCDataset, P2018Dataset,
                              SequenceDataset, SHHSDataset, SleepEDFxDataset)
from torcheeg.models import SleepDG
from torcheeg.trainers import SleepDGTrainer


SOURCES = ('sleepedf', 'hmc', 'isruc', 'shhs', 'p2018')
LABELS = {'Sleep stage W': 0, 'Sleep stage N1': 1, 'Sleep stage N2': 2,
          'Sleep stage N3': 3, 'Sleep stage R': 4}
MODEL = dict(sequence_length=20, num_electrodes=2, chunk_size=3000,
             hid_channels=512, num_heads=8, mlp_channels=512, num_layers=1,
             num_classes=5, dropout=0.1)


def subject_partitions(dataset, seed):
    subjects = sorted(dataset.info['subject_id'].unique(), key=str)
    if len(subjects) < 2:
        raise ValueError('Each source dataset needs at least two participants for source validation.')
    order = np.random.RandomState(seed).permutation(len(subjects))
    count = max(1, int(math.ceil(len(subjects) * .2)))
    validation = {subjects[i] for i in order[:count]}
    views = []
    for select in (False, True):
        view = copy(dataset)
        mask = dataset.info['subject_id'].isin(validation)
        view.info = dataset.info.loc[mask if select else ~mask].copy().reset_index(drop=True)
        views.append(view)
    return tuple(views)


def complete_epochs(dataset, wake_margin=None):
    # Apply the same deterministic selection to source and target records.
    # Keep original times so a removed interval never becomes adjacent data.
    keep, dropped = [], []
    for record, group in dataset.info.groupby('_record_id', sort=False):
        group = group.sort_values('start_at', kind='mergesort')
        if wake_margin is not None:
            sleep = group.loc[group['label'] != 'Sleep stage W']
            if sleep.empty:
                raise ValueError(f'Record {record} has no scored sleep for wake trimming.')
            begin = sleep['start_at'].iloc[0] - 30 * wake_margin
            end = sleep['end_at'].iloc[-1] + 30 * wake_margin
            group = group.loc[(group['start_at'] >= begin) & (group['end_at'] <= end)]
        cuts = np.flatnonzero(~np.isclose(group['start_at'].to_numpy()[1:],
                                         group['end_at'].to_numpy()[:-1], rtol=0, atol=1e-6)) + 1
        selected = []
        for run in np.split(group.index.to_numpy(), cuts):
            selected.extend(run[:len(run) // 20 * 20].tolist())
        if not selected:
            dropped.append(str(record))
        keep.extend(selected)
    if not keep:
        raise ValueError('No complete continuous 20-epoch sequences remain in this partition.')
    view = copy(dataset)
    view.info = dataset.info.loc[keep].copy().reset_index(drop=True)
    return view, dropped


class RecordNormalizedDataset(Dataset):
    """Fit separate channel statistics to each retained record, never across records."""

    def __init__(self, dataset):
        self.dataset = copy(dataset)
        self.info = dataset.info.copy(deep=True).reset_index(drop=True)
        self.dataset.info = self.info.copy(deep=True)
        self.statistics = {}
        for record, group in self.info.groupby('_record_id', sort=False):
            count, mean, square = 0, np.zeros(2), np.zeros(2)
            for index in group.index:
                values = np.asarray(self.dataset[int(index)][0], dtype=np.float64)
                if values.shape != (2, 3000) or not np.isfinite(values).all():
                    raise ValueError('SleepDG needs finite [2, 3000] epochs before normalization.')
                size = values.shape[1]
                local_mean = values.mean(axis=1)
                local_square = np.sum((values - local_mean[:, None]) ** 2, axis=1)
                delta = local_mean - mean
                total = count + size
                square += local_square + delta ** 2 * count * size / total
                mean += delta * size / total
                count = total
            scale = np.sqrt(square / count)
            scale[scale == 0] = 1.
            self.statistics[str(record)] = dict(mean=mean.tolist(), scale=scale.tolist(), samples=count)
        self.info['normalization'] = 'per_record_retained_epochs'

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        values, label = self.dataset[index]
        stats = self.statistics[str(self.info.iloc[index]['_record_id'])]
        values = (np.asarray(values, dtype=np.float64) - np.asarray(stats['mean'])[:, None]) / np.asarray(stats['scale'])[:, None]
        if not np.isfinite(values).all():
            raise ValueError('Normalized epochs must be finite.')
        return torch.from_numpy(values.astype(np.float32)), label


class WorkflowState(pl.Callback):
    def __init__(self, configuration):
        self.configuration = configuration

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['sleepdg_workflow'] = self.configuration


def resume_epoch(path, configuration):
    # Loading full optimizer checkpoints requires an owned, trusted file.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('sleepdg_workflow') != configuration:
        raise ValueError('The checkpoint belongs to a different workflow configuration.')
    return int(state['epoch']) + 1


class SourceBatchSampler(Sampler):
    """Use the same number of sequences from every source in each batch."""

    def __init__(self, lengths, batch_size, seed):
        if not lengths or any(length < 1 for length in lengths):
            raise ValueError('Every source needs complete training sequences.')
        if batch_size < len(lengths) or batch_size % len(lengths):
            raise ValueError('batch-size must be a positive multiple of the four source datasets.')
        self.lengths, self.batch_size, self.seed = list(lengths), batch_size, seed
        self.epoch = 0
        self.per_source = batch_size // len(lengths)
        self.offsets = np.cumsum([0] + list(lengths[:-1]))

    def __len__(self):
        return int(math.ceil(max(self.lengths) / self.per_source))

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        pools = [[] for _ in self.lengths]
        for _ in range(len(self)):
            batch = []
            for source, size in enumerate(self.lengths):
                for _ in range(self.per_source):
                    if not pools[source]:
                        pools[source] = rng.permutation(size).tolist()
                    batch.append(int(self.offsets[source]) + pools[source].pop())
            rng.shuffle(batch)
            yield batch


def source_dataset(name, root, cache):
    common = dict(root_path=root, io_path=str(cache), io_mode='pickle', num_worker=0,
                  label_transform=transforms.Compose([transforms.Select('label'), transforms.Mapping(LABELS)]))
    if name == 'sleepedf':
        return SleepEDFxDataset(**common, studies=['cassette'], channels=['EEG Fpz-Cz', 'EOG horizontal'],
                                l_freq=.3, h_freq=35., sfreq=100, missing_channels='raise')
    if name == 'hmc':
        return HMCDataset(**common, channels=['EEG F4-M1', 'EOG E1-M2'],
                          l_freq=None, h_freq=None, sfreq=100, start_at=0, end_at=None,
                          missing_channels='raise')
    if name == 'isruc':
        return ISRUCDataset(**common, groups=[0], channels=[('C3-A2', 'C3-M2'), ('LOC-A2', 'LOC-M2', 'E1-M2')],
                            l_freq=.3, h_freq=35., sfreq=100, missing_channels='raise', derive_channels=True)
    if name == 'shhs':
        return SHHSDataset(**common, visits=[1], channels=['EEG', 'EOG(L)'],
                           l_freq=.3, h_freq=35., sfreq=100)
    if name == 'p2018':
        return P2018Dataset(**common, channels=['C3-M2', 'E1-M2'], l_freq=.3, h_freq=35., sfreq=100)
    raise ValueError(f'Unknown sleep dataset: {name}.')


def prepare_sequences(dataset, name, domain=None):
    view, dropped = complete_epochs(dataset, wake_margin=60 if name == 'sleepedf' else None)
    normalized = RecordNormalizedDataset(view)
    sequences = SequenceDataset(normalized, sequence_length=20, stride=20, domain_id=domain)
    metadata = dict(epochs=len(view), sequences=len(sequences), dropped_records=dropped,
                    subjects=sorted(str(value) for value in view.info['subject_id'].unique()),
                    records=sorted(str(value) for value in view.info['record_id'].unique()),
                    normalization=normalized.statistics)
    return sequences, metadata


def main():
    parser = argparse.ArgumentParser(description='Train native SleepDG on four sources and test one held-out dataset.')
    for name in SOURCES:
        parser.add_argument(f'--{name}-root', required=True)
    parser.add_argument('--target', choices=SOURCES, default='shhs')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--accelerator', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--scheduler-steps', type=int, default=None)
    for phase in ('train', 'val', 'test'):
        parser.add_argument(f'--{phase}-batches', type=int, default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 4 or args.batch_size % 4 or args.seed < 0:
        parser.error('epochs must be positive, batch-size a multiple of four, and seed nonnegative.')
    for phase in ('train', 'val', 'test'):
        limit = getattr(args, phase + '_batches')
        if limit is not None and limit < 1:
            parser.error('Batch limits must be positive integers.')
    if args.scheduler_steps is not None and args.scheduler_steps < 1:
        parser.error('scheduler-steps must be positive.')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run.json').exists() and args.resume is None:
        raise ValueError('Choose a fresh output directory, or explicitly resume its trusted checkpoint.')
    pl.seed_everything(args.seed, workers=True)
    sources = [name for name in SOURCES if name != args.target]
    domains = {name: index for index, name in enumerate(sources)}
    training, validation, partitions = [], [], {}
    for name in SOURCES:
        dataset = source_dataset(name, getattr(args, name + '_root'), output / 'cache' / name)
        if name == args.target:
            target, partitions[name] = prepare_sequences(dataset, name)
        else:
            train, valid = subject_partitions(dataset, args.seed)
            train_sequences, train_info = prepare_sequences(train, name, domains[name])
            val_sequences, val_info = prepare_sequences(valid, name)
            training.append(train_sequences)
            validation.append(val_sequences)
            partitions[name] = dict(train=train_info, validation=val_info)
    sampler = SourceBatchSampler([len(dataset) for dataset in training], args.batch_size, args.seed)
    limits = {phase: getattr(args, phase + '_batches') for phase in ('train', 'val', 'test')}
    steps_per_epoch = min(len(sampler), args.train_batches) if args.train_batches else len(sampler)
    horizon = args.scheduler_steps if args.scheduler_steps is not None else args.epochs * steps_per_epoch
    configuration = dict(protocol='torcheeg-sleepdg-five-source-v2', target=args.target, source_domains=domains, partitions=partitions,
                         model_config=MODEL, seed=args.seed, batch_size=args.batch_size,
                         batch_limits=limits, scheduler_steps=horizon,
                         roots={name: str(Path(getattr(args, name + '_root')).resolve()) for name in SOURCES})
    manifest = output / 'run.json'
    if args.resume is not None:
        if not manifest.exists() or json.loads(manifest.read_text()) != configuration:
            raise ValueError('Resume needs the same recorded data, partitions, batch settings and schedule horizon.')
        sampler.epoch = resume_epoch(args.resume, configuration)
    else:
        manifest.write_text(json.dumps(configuration, indent=2))
    train_loader = DataLoader(ConcatDataset(training), batch_sampler=sampler, num_workers=0)
    val_loader = DataLoader(ConcatDataset(validation), batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(target, batch_size=args.batch_size, num_workers=0)
    model = SleepDG(**MODEL)
    trainer = SleepDGTrainer(model, num_domains=4, lr=.001, weight_decay=.0001,
                             scheduler_steps=horizon, accelerator=args.accelerator,
                             devices=1, metrics=['accuracy', 'f1score'])
    checkpoint = ModelCheckpoint(dirpath=str(output / 'checkpoints'), monitor='val_accuracy',
                                 mode='max', save_top_k=1, save_last=True, filename='source-{epoch:03d}')
    fit_options = dict(logger=False, callbacks=[checkpoint, WorkflowState(configuration)], num_sanity_val_steps=0,
                       default_root_dir=str(output), ckpt_path=args.resume)
    for phase in ('train', 'val'):
        limit = getattr(args, phase + '_batches')
        if limit is not None:
            fit_options[f'limit_{phase}_batches'] = limit
    trainer.fit(train_loader, val_loader, max_epochs=args.epochs, **fit_options)
    completed_steps = int(trainer.trainer.global_step)
    selected = checkpoint.best_model_path
    if not selected:
        raise RuntimeError('Source validation did not produce a selected checkpoint.')
    test_options = dict(logger=False, enable_checkpointing=False, ckpt_path=selected)
    if args.test_batches is not None:
        test_options['limit_test_batches'] = args.test_batches
    metrics = trainer.test(test_loader, **test_options)[0]
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('Target evaluation returned nonfinite metrics.')
    state = torch.load(selected, map_location='cpu', weights_only=False)
    restored = SleepDG(**MODEL).eval()
    restored.load_state_dict({key[len('model.'):]: value for key, value in state['state_dict'].items()
                              if key.startswith('model.')}, strict=True)
    # Check the exported model against the selected checkpoint, not a later epoch.
    trainer.model.eval().cpu()
    probe = next(iter(test_loader))[0][:1]
    with torch.no_grad():
        torch.testing.assert_close(restored(probe), trainer.model(probe), rtol=0, atol=0)
    torch.save(dict(model_config_json=json.dumps(MODEL), state_dict=restored.state_dict(), target=args.target,
                    source_domains=domains, checkpoint=selected), output / 'model.pt')
    # JSON configuration also works with the older weights-only unpickler,
    # which cannot read Python floating metadata on Torch 1.13.
    exported = torch.load(output / 'model.pt', map_location='cpu', weights_only=True)
    roundtrip = SleepDG(**json.loads(exported['model_config_json'])).eval()
    roundtrip.load_state_dict(exported['state_dict'], strict=True)
    with torch.no_grad():
        torch.testing.assert_close(roundtrip(probe), restored(probe), rtol=0, atol=0)
    report = dict(protocol='TorchEEG corrected five-source workflow; not published-score reproduction',
                  target=args.target, source_domains=domains, partitions=partitions,
                  normalization='retained-record statistics, including unlabeled target records',
                  model_config=MODEL, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                  batch_limits=limits, scheduler_steps=horizon,
                  selected_checkpoint=selected, last_checkpoint=checkpoint.last_model_path,
                  selected_epoch=int(state['epoch']), selected_optimizer_steps=int(state['global_step']),
                  training_optimizer_steps=completed_steps,
                  target_metrics=metrics, exported_model='model.pt', strict_reload_equal=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(target=args.target, checkpoint=selected, metrics=metrics,
                          report=str(output / 'report.json'))))


if __name__ == '__main__':
    main()
