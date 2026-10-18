"""
Train EEGNet on ISRUC
==========================

Use the original ISRUC-Sleep REC/text pairs from
https://sleeptight.isr.uc.pt/. Cite Khalighi et al. (2016) and the provider.
Do not substitute the separately prepared MATLAB arrays.

Select recorded F4-A1/F4-M1 EEG and LOC-A2/LOC-M2 EOG candidates. When only
individual electrode/reference channels exist, compute their actual bipolar
difference. Keep the recorded reference names and provenance; A and M names
are not declared equivalent. Missing or ambiguous channels fail.

Use 0.5--30 Hz FIR filtering, 100 Hz output and complete 30-second epochs
in volts, then normalize each epoch per channel. Hold out the final sorted
participant, including both Subgroup_2 nights, and train for a fixed epoch
count without test-based parameter selection. This is a standalone workflow,
not a published sleep-staging benchmark.

Install the source and run::

    pip install -e ".[dataset,trainer]"
    python examples/examples_isruc_eegnet.py --root-path ./ISRUC-SLEEP \
        --groups 0 --scorer 1 --epochs 1 --output-path ./isruc-eegnet.pt

Subgroup codes 0, 1 and 2 select Subgroup_1, Subgroup_2 and Subgroup_3.
The default uses Subgroup_1 and expert 1. Select expert 2 explicitly; there
is no consensus or fallback to another score file. At least two participants
are required. A scoring-count mismatch raises an error.

Use new preprocessing and split caches after changing source membership,
references, scorer or preprocessing. The exported model contains complete
constructor settings and reader/reference provenance as JSON, with a sidecar
report of the actual split and evaluated scope. The gallery does not execute
training.
"""

import argparse
import json
import math
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import ISRUCDataset
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import EEGNet
from torcheeg.trainers import ClassifierTrainer


MODEL_CONFIG = dict(num_electrodes=2, chunk_size=3000, num_classes=5,
                    F1=8, F2=16, D=2, kernel_1=64, kernel_2=16, dropout=.25)
CHANNELS = [('F4-A1', 'F4-M1'), ('LOC-A2', 'LOC-M2')]
LABELS = {'Sleep stage W': 0, 'Sleep stage N1': 1, 'Sleep stage N2': 2,
          'Sleep stage N3': 3, 'Sleep stage R': 4}


def channel_provenance(dataset):
    records = {}
    for record, rows in dataset.info.groupby('record_id', sort=False):
        fields = ('channel_names', 'derived_channels', 'zero_filled_channels')
        if any(rows[field].nunique() != 1 for field in fields):
            raise ValueError('Channel provenance must be constant within each record.')
        values = {field: json.loads(rows.iloc[0][field]) for field in fields}
        if len(values['channel_names']) != 2 or values['zero_filled_channels']:
            raise ValueError('Require two measured or explicitly derived ISRUC channels.')
        records[str(record)] = values
    return records


def main():
    parser = argparse.ArgumentParser(description='Train EEGNet on ISRUC with one held-out participant, including both nights.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./isruc-eeg-eog-100hz-30s')
    parser.add_argument('--output-path', default='./isruc-eegnet.pt')
    parser.add_argument('--groups', nargs='+', type=int, choices=[0, 1, 2], default=[0])
    parser.add_argument('--scorer', type=int, choices=[1, 2], default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs must be positive and batch-size must be at least two.')
    output = Path(args.output_path)
    report_path = output.with_suffix(output.suffix + '.json')
    if output.exists() or report_path.exists():
        raise ValueError('Choose a fresh output path for this training run.')
    pl.seed_everything(42, workers=True)
    dataset = ISRUCDataset(
        root_path=args.root_path, io_path=args.cache_path, io_mode='pickle', num_worker=0,
        channels=CHANNELS, sfreq=100, l_freq=.5, h_freq=30.,
        groups=args.groups, scorer=args.scorer, missing_channels='raise', derive_channels=True,
        online_transform=transforms.Compose([
            transforms.MeanStdNormalize(axis=-1), transforms.ToTensor(), transforms.To2d()]),
        label_transform=transforms.Compose([transforms.Select('label'), transforms.Mapping(LABELS)]))
    groups = sorted(dataset.info['subject_id'].unique(), key=str)
    if len(groups) < 2:
        raise ValueError('At least two ISRUC participants with complete selected scorer files are required.')
    splitter = LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split')
    train, test = next(splitter.split(dataset, subject=groups[-1]))
    record_channels = channel_provenance(dataset)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=0)
    model = EEGNet(**MODEL_CONFIG)
    trainer = ClassifierTrainer(model=model, num_classes=5, lr=.001, accelerator='cpu',
                                devices=1, metrics=['accuracy', 'f1score'])
    options = dict(logger=False, enable_checkpointing=False, enable_progress_bar=False,
                   enable_model_summary=False)
    trainer.fit(train_loader, None, max_epochs=args.epochs, num_sanity_val_steps=0,
                default_root_dir=str(output.parent / 'isruc-training'), **options)
    steps = int(trainer.trainer.global_step)
    metrics = trainer.test(test_loader, **options)[0]
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('ISRUC evaluation returned nonfinite metrics.')
    # Freeze constrained convolution weights for a stable exported inference state.
    model.requires_grad_(False).eval()
    probe = next(iter(test_loader))[0][:2]
    restored = EEGNet(**MODEL_CONFIG).requires_grad_(False).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(probe), model(probe), rtol=0, atol=0)
    preprocessing = dict(channels=CHANNELS, signal_unit='V', sfreq=100, epoch_seconds=30,
                         l_freq=.5, h_freq=30., groups=args.groups, scorer=args.scorer,
                         derive_channels=True, record_channels=record_channels,
                         normalization='per_epoch_per_channel', label_mapping=LABELS)
    partitions = dict(train_subjects=sorted(str(value) for value in train.info.subject_id.unique()),
                      test_subjects=sorted(str(value) for value in test.info.subject_id.unique()))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), model_config_json=json.dumps(MODEL_CONFIG),
                    preprocessing_json=json.dumps(preprocessing), partitions_json=json.dumps(partitions)), output)
    payload = torch.load(output, map_location='cpu', weights_only=True)
    roundtrip = EEGNet(**json.loads(payload['model_config_json'])).requires_grad_(False).eval()
    roundtrip.load_state_dict(payload['state_dict'], strict=True)
    with torch.no_grad():
        torch.testing.assert_close(roundtrip(probe), model(probe), rtol=0, atol=0)
    report = dict(model_config=MODEL_CONFIG, preprocessing=preprocessing, partitions=partitions,
                  epochs=args.epochs, batch_size=args.batch_size, seed=42, optimizer_steps=steps,
                  train_epochs=len(train), test_epochs=len(test), test_metrics=metrics,
                  strict_reload_equal=True, output_path=str(output),
                  train_records=sorted(str(value) for value in train.info.record_id.unique()),
                  test_records=sorted(str(value) for value in test.info.record_id.unique()),
                  scope='Participant holdout with both Subgroup_2 nights kept together; no published benchmark claim.')
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(test_subject=str(groups[-1]), metrics=metrics, report=str(report_path))))


if __name__ == '__main__':
    main()
