"""
Train ADFSU EEGNet
==================

Read the OSF heaw6 version-1 resting EEG archive through ADFSUDataset.
Cite Vicchietti et al., Scientific Reports 13, 8184 (2023):
https://www.nature.com/articles/s41598-023-32664-8.

Use one condition per run: Eyes_closed by default, or Eyes_open explicitly.
The release has no cross-condition clinical identity linkage table. Hold
out the highest natural-numbered source folder in each diagnosis class,
keeping all its windows together. The grouping key is group_PacienteN;
it does not verify a clinical identity across conditions. Require both
classes and at least two folder identities per class. This single fixed
split measures pooled window metrics, not clinical validity or a published
benchmark. Train for a fixed epoch count without test-based selection.

Keep all 19 canonical electrodes and native 128 Hz. The source has already
been band limited to 0.5--30 Hz and cleaned by its provider. Add no filter,
reference or resampling. Source calibration is unspecified: preserve source
numbers, then normalize each nonoverlapping one-second window per channel.
Do not fit statistics across windows or evaluation groups. Healthy is 0,
probable AD is 1; resting condition is not the diagnosis target.

Install the source and run::

    pip install -e ".[dataset,trainer]"
    python examples/examples_adfsu_eegnet.py --root-path ./EEG_data --state Eyes_closed --epochs 1 --batch-size 32 --output-path ./adfsu-eyes-closed.pt

Choose a separate output for Eyes_open. The selected state is appended to
the cache root, preventing reuse of the other condition's cache. Use a new
cache root when source files or preprocessing change. The reader warns and
omits the archive's completely empty Eyes_open/Paciente5 condition; partial
channel corruption is an error.

Export all model settings, tensor states, preprocessing and actual folder/
record partitions as safe tensor/JSON data, with a separate JSON report.
Frozen constrained convolutions allow exact inference restoration. Existing
model/report files are preserved. The gallery renders this example without
running the training workflow.
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
from torcheeg.datasets import ADFSUDataset
from torcheeg.datasets.constants import ADFSU_CHANNEL_LIST
from torcheeg.models import EEGNet
from torcheeg.trainers import ClassifierTrainer


MODEL_CONFIG = dict(num_electrodes=19, chunk_size=128, num_classes=2,
                    F1=8, F2=16, D=2, kernel_1=64, kernel_2=16, dropout=.25)
CHANNELS = ADFSU_CHANNEL_LIST
LABELS = {'Healthy': 0, 'AD': 1}


def main():
    parser = argparse.ArgumentParser(description='Train ADFSU EEGNet with one held-out source folder per diagnosis class in one resting condition.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./adfsu-native-19ch-128hz-1s')
    parser.add_argument('--state', choices=['Eyes_closed', 'Eyes_open'], default='Eyes_closed')
    parser.add_argument('--output-path', default='./adfsu-eegnet.pt')
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
    dataset = ADFSUDataset(
        root_path=args.root_path, io_path=str(Path(args.cache_path) / args.state), io_mode='pickle', num_worker=0,
        sfreq=128, chunk_size=128, overlap=0, num_channel=19,
        states=(args.state,), notch_freqs=None, l_freq=None, h_freq=None, ref_channels=None,
        online_transform=transforms.Compose([
            transforms.MeanStdNormalize(axis=-1), transforms.ToTensor(), transforms.To2d()]),
        label_transform=transforms.Select('label'))
    if dataset.info.groupby('subject_id')['label'].nunique().ne(1).any():
        raise ValueError('Each source folder must have one diagnosis label across its records.')
    people = dataset.info[['subject_id', 'label']].drop_duplicates()
    if set(people['label']) != {0, 1}:
        raise ValueError('Require both diagnosis classes before creating this holdout.')
    test_by_label = {}
    for label in range(2):
        subjects = sorted((str(value) for value in people.loc[people['label'] == label, 'subject_id']),
                          key=lambda value: int(value.rsplit('Paciente', 1)[1]))
        if len(subjects) < 2:
            raise ValueError('Require at least two source folders per diagnosis class for disjoint train/test data.')
        test_by_label[str(label)] = subjects[-1]
    train, test = copy.copy(dataset), copy.copy(dataset)
    selected = dataset.info['subject_id'].isin(test_by_label.values())
    train.info = dataset.info.loc[~selected].copy().reset_index(drop=True)
    test.info = dataset.info.loc[selected].copy().reset_index(drop=True)
    if set(train.info['subject_id']) & set(test.info['subject_id']):
        raise ValueError('Training and evaluation source folders must be disjoint.')
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=0)
    model = EEGNet(**MODEL_CONFIG)
    trainer = ClassifierTrainer(model=model, num_classes=2, lr=.001, accelerator='cpu',
                                devices=1, metrics=['accuracy', 'f1score'])
    options = dict(logger=False, enable_checkpointing=False, enable_progress_bar=False,
                   enable_model_summary=False)
    trainer.fit(train_loader, None, max_epochs=args.epochs, num_sanity_val_steps=0,
                default_root_dir=str(output.parent / 'adfsu-training'), **options)
    steps = int(trainer.trainer.global_step)
    metrics = trainer.test(test_loader, **options)[0]
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('ADFSU evaluation returned nonfinite metrics.')
    # Freeze constrained convolution weights for a stable exported inference state.
    model.requires_grad_(False).eval()
    probe = next(iter(test_loader))[0][:2]
    restored = EEGNet(**MODEL_CONFIG).requires_grad_(False).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(probe), model(probe), rtol=0, atol=0)
    preprocessing = dict(channels=CHANNELS, signal_unit='source',
                         units_basis='source_archive_has_no_physical_calibration',
                         sfreq=128, chunk_size=128, overlap=0, num_channel=19,
                         states=[args.state], notch_freqs=None,
                         l_freq=None, h_freq=None, ref_channels=None,
                         provider_preprocessing='released_0.5_to_30_Hz_band_limit_and_technician_artifact_removal',
                         normalization='per_window_per_channel',
                         constant_channels='equal_extrema_centered_exactly', source_label_mapping=LABELS)
    partitions = dict(selection='last_natural_folder_per_diagnosis_within_one_condition',
                      identity_basis='group_and_source_folder_single_condition',
                      cross_condition_clinical_linkage_verified=False,
                      test_groups_by_label=test_by_label,
                      group_labels={str(row.subject_id): int(row.label) for row in people.itertuples(index=False)},
                      train_subject_groups=sorted(str(value) for value in train.info.subject_id.unique()),
                      test_subject_groups=sorted(str(value) for value in test.info.subject_id.unique()),
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
                  train_condition_records=int(train.info.record_id.nunique()), test_condition_records=int(test.info.record_id.nunique()),
                  train_condition_windows={str(key): int(value) for key, value in train.info.state.value_counts().items()},
                  test_condition_windows={str(key): int(value) for key, value in test.info.state.value_counts().items()},
                  test_metrics=metrics,
                  strict_reload_equal=True, output_path=str(output),
                  scope='One fixed source-folder holdout per diagnosis class within one condition; pooled window-level metrics; no clinical or published benchmark claim.')
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(test_subject_groups=partitions['test_subject_groups'], metrics=metrics, report=str(report_path))))


if __name__ == '__main__':
    main()
