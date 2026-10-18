"""
Train photostimulation ADFTD EEGNet
===================================

Read OpenNeuro ds006036 v1.0.5 with ADFTDPSDataset. Cite Ntetska et al.,
Data 10(5), 64 (2025), https://doi.org/10.3390/data10050064. Predict diagnosis:
healthy controls (C, 0), Alzheimer's disease (A, 1), and frontotemporal
dementia (F, 2). PHOTO and Photo/HV markers are acquisition events, not the
targets. This example does not infer per-window stimulation frequency or
an SSVEP protocol. The released task is photomark; no per-window eye state
is inferred from the source's inconsistent descriptive wording.

Keep the full 19-channel canonical order at native 500 Hz. The sidecar unit
fields are n/a. Convert EEGLAB microvolt numbers to volts by the format
convention; this is not independent hardware calibration. Add no filtering,
reference or resampling. Normalize each nonoverlapping one-second window
per channel, without statistics fitted across windows or participants.

Hold out the highest numeric participant ID within each diagnosis. Require
at least two participants per diagnosis, and keep all sessions, runs and
windows for a participant together. Train a fixed number of epochs without
test-based selection. Pooled window metrics for one fixed split do not
establish clinical validity or reproduce a published benchmark. Resting-state
ds004504 shares these participant IDs: do not treat equal IDs across tasks
as independent training and test people.

Install the source and run::

    pip install -e ".[dataset,trainer]"
    python examples/examples_adftd_ps_eegnet.py --root-path ./ADFTD-PS --epochs 1 --batch-size 32 --output-path ./adftd-ps-eegnet.pt

The default selects original released SET files. Pass --preprocessed to
read provider-cleaned derivatives/eeglab files, with no extra preprocessing.
Both modes preserve stored segment boundaries: windows never cross removed
ASR intervals. The source mode is appended to the cache root. Use a new cache
root when source files or settings change and a separate model/report output.

Save all nine EEGNet settings, tensor states, preprocessing and actual
participant/record partitions as safe tensor/JSON data. Restore with
weights_only=True and strict state loading. Frozen constrained convolution
weights allow exact inference restoration. Existing model/report files are
preserved. The gallery renders this workflow without running training.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import ADFTDPSDataset
from torcheeg.datasets.constants import ADFTD_PS_CHANNEL_LIST
from torcheeg.models import EEGNet
from torcheeg.trainers import ClassifierTrainer


MODEL_CONFIG = dict(num_electrodes=19, chunk_size=500, num_classes=3,
                    F1=8, F2=16, D=2, kernel_1=64, kernel_2=16, dropout=.25)
CHANNELS = ADFTD_PS_CHANNEL_LIST
LABELS = {'C': 0, 'A': 1, 'F': 2}


def main():
    parser = argparse.ArgumentParser(description='Train photostimulation ADFTD EEGNet with one held-out participant per diagnosis class.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./adftd-ps-native-19ch-500hz-1s')
    parser.add_argument('--preprocessed', action='store_true', help='Read provider-cleaned derivatives instead of the original release.')
    parser.add_argument('--output-path', default='./adftd-ps-eegnet.pt')
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
    dataset = ADFTDPSDataset(
        root_path=args.root_path, io_path=str(Path(args.cache_path) / ('preprocessed' if args.preprocessed else 'raw')), io_mode='pickle', num_worker=0,
        sfreq=500, chunk_size=500, overlap=0, num_channel=19,
        preprocessed=args.preprocessed, notch_freqs=None, l_freq=None, h_freq=None, ref_channels=None,
        online_transform=transforms.Compose([
            transforms.MeanStdNormalize(axis=-1), transforms.ToTensor(), transforms.To2d()]),
        label_transform=transforms.Select('label'))
    if dataset.info.groupby('subject_id')['label'].nunique().ne(1).any():
        raise ValueError('Each participant must have one diagnosis across all records.')
    if dataset.info.groupby('subject_id')['participant_id'].nunique().ne(1).any():
        raise ValueError('Each numeric participant ID must identify one BIDS participant.')
    people = dataset.info[['subject_id', 'participant_id', 'label']].drop_duplicates()
    if set(people['label']) != {0, 1, 2}:
        raise ValueError('Require all three diagnosis classes before creating this holdout.')
    test_by_label = {}
    for label in range(3):
        subjects = sorted(int(value) for value in people.loc[people['label'] == label, 'subject_id'])
        if len(subjects) < 2:
            raise ValueError('Require at least two participants per diagnosis for disjoint train/test data.')
        test_by_label[str(label)] = subjects[-1]
    train, test = copy.copy(dataset), copy.copy(dataset)
    selected = dataset.info['subject_id'].isin(test_by_label.values())
    train.info = dataset.info.loc[~selected].copy().reset_index(drop=True)
    test.info = dataset.info.loc[selected].copy().reset_index(drop=True)
    if set(train.info['subject_id']) & set(test.info['subject_id']):
        raise ValueError('Training and evaluation participants must be disjoint.')
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=0)
    model = EEGNet(**MODEL_CONFIG)
    trainer = ClassifierTrainer(model=model, num_classes=3, lr=.001, accelerator='cpu',
                                devices=1, metrics=['accuracy', 'f1score'])
    options = dict(logger=False, enable_checkpointing=False, enable_progress_bar=False,
                   enable_model_summary=False)
    trainer.fit(train_loader, None, max_epochs=args.epochs, num_sanity_val_steps=0,
                default_root_dir=str(output.parent / 'adftd-ps-training'), **options)
    steps = int(trainer.trainer.global_step)
    metrics = trainer.test(test_loader, **options)[0]
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('ADFTD_PS evaluation returned nonfinite metrics.')
    # Freeze constrained convolution weights for a stable exported inference state.
    model.requires_grad_(False).eval()
    probe = next(iter(test_loader))[0][:2]
    restored = EEGNet(**MODEL_CONFIG).requires_grad_(False).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(probe), model(probe), rtol=0, atol=0)
    preprocessing = dict(channels=CHANNELS, signal_unit='V',
                         units_basis='EEGLAB_microvolt_format_convention', task='photomark',
                         sfreq=500, chunk_size=500, overlap=0, num_channel=19,
                         data_source='preprocessed' if args.preprocessed else 'raw',
                         notch_freqs=None, l_freq=None, h_freq=None, ref_channels=None,
                         boundary_policy='window_within_each_stored_contiguous_segment',
                         normalization='per_window_per_channel',
                         constant_channels='equal_extrema_centered_exactly', source_label_mapping=LABELS)
    partitions = dict(selection='highest_numeric_participant_per_diagnosis',
                      identity_basis='BIDS_participant_id_within_ds006036',
                      cross_task_ids_match_ds004504=True,
                      test_subject_ids_by_label=test_by_label,
                      group_labels={str(row.participant_id): int(row.label) for row in people.itertuples(index=False)},
                      train_subject_ids=sorted(int(value) for value in train.info.subject_id.unique()),
                      test_subject_ids=sorted(int(value) for value in test.info.subject_id.unique()),
                      train_participants=sorted(str(value) for value in train.info.participant_id.unique()),
                      test_participants=sorted(str(value) for value in test.info.participant_id.unique()),
                      train_records=sorted(str(value) for value in train.info.record_id.unique()),
                      test_records=sorted(str(value) for value in test.info.record_id.unique()))
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
                  train_windows=len(train), test_windows=len(test),
                  train_records=int(train.info.record_id.nunique()), test_records=int(test.info.record_id.nunique()),
                  test_metrics=metrics,
                  strict_reload_equal=True, output_path=str(output),
                  scope='One fixed participant holdout per diagnosis class; pooled window-level metrics; no clinical or published benchmark claim.')
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(test_participants=partitions['test_participants'], metrics=metrics, report=str(report_path))))


if __name__ == '__main__':
    main()
