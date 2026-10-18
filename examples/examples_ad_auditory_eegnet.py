"""
Train AD-Auditory EEGNet
========================

Read OpenNeuro ds005048 v1.0.0 through ADAuditoryDataset. Cite Lahijanian,
Aghajan and Vahabi (Scientific Reports 14, 13153, 2024), and the dataset DOI:
https://openneuro.org/datasets/ds005048/versions/1.0.0.

The source contains provider-cleaned EEG. Keep the 19 EEG channels, native
250 Hz and nonoverlapping one-second windows from both Stimulus and Rest.
Add no filter, reference or resampling. Normalize each window per channel;
fit no statistics across windows or people. Values follow the EEGLAB
microvolt convention; source channel-unit metadata are n/a. Diagnosis
labels are HC 0, Mild/Moderate AD together 1, and MCI 2. Unknown diagnoses
are excluded by the reader. Condition values are not the diagnosis target.

Within each diagnosis class, hold out its last sorted participant ID and
train on the others. Require all three classes and at least two participants
in each. All records, conditions and windows from a person stay together.
Record the actual split and window/event counts. This is one fixed split,
with pooled window-level metrics; it is not a full clinical evaluation or
published benchmark protocol. Training uses a fixed epoch count; test
participants do not select model settings or epochs.

Install the source and run::

    pip install -e ".[dataset,trainer]"
    python examples/examples_ad_auditory_eegnet.py --root-path ./AD-Auditory --epochs 1 --batch-size 64 --output-path ./ad-auditory-eegnet.pt

The complete constructor, tensor state, preprocessing and participant/record
partitions are exported with a JSON report. Frozen constrained convolutions
allow exact inference restoration. Existing model and report files are
preserved. Use a fresh cache when sources, conditions or preprocessing change.
The gallery renders this example without executing the training workflow.
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
from torcheeg.datasets import ADAuditoryDataset
from torcheeg.datasets.constants import AD_AUDITORY_CHANNEL_LIST
from torcheeg.models import EEGNet
from torcheeg.trainers import ClassifierTrainer


MODEL_CONFIG = dict(num_electrodes=19, chunk_size=250, num_classes=3,
                    F1=8, F2=16, D=2, kernel_1=64, kernel_2=16, dropout=.25)
CHANNELS = AD_AUDITORY_CHANNEL_LIST
LABELS = {'Normal': 0, 'Mild AD': 1, 'Moderate AD': 1, 'MCI': 2}


def main():
    parser = argparse.ArgumentParser(description='Train AD-Auditory EEGNet with one held-out participant per diagnosis class.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./ad-auditory-native-19ch-250hz-1s')
    parser.add_argument('--output-path', default='./ad-auditory-eegnet.pt')
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
    dataset = ADAuditoryDataset(
        root_path=args.root_path, io_path=args.cache_path, io_mode='pickle', num_worker=0,
        sfreq=250, chunk_size=250, overlap=0, num_channel=19,
        trial_types=('Stimulus', 'Rest'), notch_freqs=None, l_freq=None, h_freq=None, ref_channels=None,
        online_transform=transforms.Compose([
            transforms.MeanStdNormalize(axis=-1), transforms.ToTensor(), transforms.To2d()]),
        label_transform=transforms.Select('label'))
    if dataset.info.groupby('subject_id')['label'].nunique().ne(1).any():
        raise ValueError('Each participant must have one diagnosis label across all records.')
    people = dataset.info[['subject_id', 'label']].drop_duplicates()
    if set(people['label']) != {0, 1, 2}:
        raise ValueError('Require all three diagnosis classes before creating this holdout.')
    test_by_label = {}
    for label in range(3):
        subjects = sorted(str(value) for value in people.loc[people['label'] == label, 'subject_id'])
        if len(subjects) < 2:
            raise ValueError('Require at least two participants per diagnosis class for disjoint train/test data.')
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
                default_root_dir=str(output.parent / 'ad-auditory-training'), **options)
    steps = int(trainer.trainer.global_step)
    metrics = trainer.test(test_loader, **options)[0]
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('AD-Auditory evaluation returned nonfinite metrics.')
    # Freeze constrained convolution weights for a stable exported inference state.
    model.requires_grad_(False).eval()
    probe = next(iter(test_loader))[0][:2]
    restored = EEGNet(**MODEL_CONFIG).requires_grad_(False).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(probe), model(probe), rtol=0, atol=0)
    preprocessing = dict(channels=CHANNELS, signal_unit='uV',
                         units_basis='EEGLAB_external_data_convention_source_channels_tsv_n_a',
                         sfreq=250, chunk_size=250, overlap=0, num_channel=19,
                         trial_types=['Stimulus', 'Rest'], notch_freqs=None,
                         l_freq=None, h_freq=None, ref_channels=None,
                         provider_preprocessing='released_cleaned_EEGLAB_recordings_ASR_ICA_and_average_reference',
                         normalization='per_window_per_channel',
                         constant_channels='equal_extrema_centered_exactly', source_label_mapping=LABELS)
    partitions = dict(selection='last_sorted_subject_per_diagnosis_class',
                      test_subjects_by_label=test_by_label,
                      subject_labels={str(row.subject_id): int(row.label) for row in people.itertuples(index=False)},
                      train_subjects=sorted(str(value) for value in train.info.subject_id.unique()),
                      test_subjects=sorted(str(value) for value in test.info.subject_id.unique()),
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
                  train_events=int(train.info.trial_id.nunique()), test_events=int(test.info.trial_id.nunique()),
                  train_condition_windows={str(key): int(value) for key, value in train.info.trial_type.value_counts().items()},
                  test_condition_windows={str(key): int(value) for key, value in test.info.trial_type.value_counts().items()},
                  test_metrics=metrics,
                  strict_reload_equal=True, output_path=str(output),
                  scope='One fixed participant holdout per diagnosis class; pooled window-level metrics; no clinical or published benchmark claim.')
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(test_subjects=partitions['test_subjects'], metrics=metrics, report=str(report_path))))


if __name__ == '__main__':
    main()
