"""
Train EEGNet on P2018 sleep stages
==================================

Use labeled training records from the PhysioNet/CinC Challenge 2018 dataset:
https://physionet.org/content/challenge-2018/1.0.0/. Cite Ghassemi et al.
(2018) and PhysioNet. Keep matching MAT, HEA and WFDB ``.arousal`` files in
each record directory below ``training``. This example predicts five sleep
stages, not the challenge's binary arousal target.

Select C3-M2 EEG and E1-M2 EOG, resample each whole signal to 100 Hz with
WFDB's FFT path, and apply an eighth-order zero-phase 0.5--30 Hz Butterworth
filter in second-order sections. Preserve native physical units from the
WFDB header (normally microvolts), then normalize each complete 30-second
epoch per channel. Store actual channel names and units per record in the
export. No implicit conversion to MNE-style volts is applied.

The first sleep-stage annotation anchors the scoring grid, even when an
arousal event occurs earlier. Exclude undefined and mixed-stage windows;
retain all complete scored wake epochs and the original time gaps. Hold out
the final sorted released record, and train for a fixed epoch count without
using test metrics for model selection. This is a holdout within the labeled
training data, not evaluation on the provider's hidden test set.

Install the source and run::

    pip install -e ".[dataset,trainer]"
    python examples/examples_p2018_eegnet.py --root-path ./P2018/training \
        --epochs 1 --output-path ./p2018-eegnet.pt

At least two labeled records are required. Use fresh preprocessing and split
caches after changing source membership, channel order, units or processing.
The full model constructor, tensor state, preprocessing and partition IDs
are exported with a JSON report. This workflow does not establish a published
sleep-staging benchmark or challenge arousal score. The gallery does not
execute training.
"""

import argparse
import json
import math
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import P2018Dataset
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import EEGNet
from torcheeg.trainers import ClassifierTrainer


MODEL_CONFIG = dict(num_electrodes=2, chunk_size=3000, num_classes=5,
                    F1=8, F2=16, D=2, kernel_1=64, kernel_2=16, dropout=.25)
CHANNELS = ['C3-M2', 'E1-M2']
LABELS = {'Sleep stage W': 0, 'Sleep stage N1': 1, 'Sleep stage N2': 2,
          'Sleep stage N3': 3, 'Sleep stage R': 4}


def channel_provenance(dataset):
    records = {}
    for record, rows in dataset.info.groupby('record_id', sort=False):
        fields = ('channel_names', 'channel_units')
        if any(rows[field].nunique() != 1 for field in fields):
            raise ValueError('Channel names and units must be constant within each record.')
        values = {field: json.loads(rows.iloc[0][field]) for field in fields}
        units = values['channel_units']
        if values['channel_names'] != CHANNELS or len(units) != 2:
            raise ValueError('Require C3-M2 and E1-M2 with their physical units in that order.')
        if any(not isinstance(unit, str) or not unit.strip() for unit in units):
            raise ValueError('Each selected channel must have a recorded physical unit.')
        records[str(record)] = values
    return records


def main():
    parser = argparse.ArgumentParser(description='Train EEGNet on P2018 with one held-out released record.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./p2018-eeg-eog-100hz-30s')
    parser.add_argument('--output-path', default='./p2018-eegnet.pt')
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
    dataset = P2018Dataset(
        root_path=args.root_path, io_path=args.cache_path, io_mode='pickle', num_worker=0,
        channels=CHANNELS, sfreq=100, l_freq=.5, h_freq=30.,
        online_transform=transforms.Compose([
            transforms.MeanStdNormalize(axis=-1), transforms.ToTensor(), transforms.To2d()]),
        label_transform=transforms.Compose([transforms.Select('label'), transforms.Mapping(LABELS)]))
    provenance = channel_provenance(dataset)
    groups = sorted(dataset.info['subject_id'].unique(), key=str)
    if len(groups) < 2:
        raise ValueError('At least two complete released records are required.')
    splitter = LeaveOneSubjectOut(split_path=args.cache_path + '-record-split')
    train, test = next(splitter.split(dataset, subject=groups[-1]))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=0)
    model = EEGNet(**MODEL_CONFIG)
    trainer = ClassifierTrainer(model=model, num_classes=5, lr=.001, accelerator='cpu',
                                devices=1, metrics=['accuracy', 'f1score'])
    options = dict(logger=False, enable_checkpointing=False, enable_progress_bar=False,
                   enable_model_summary=False)
    trainer.fit(train_loader, None, max_epochs=args.epochs, num_sanity_val_steps=0,
                default_root_dir=str(output.parent / 'p2018-training'), **options)
    steps = int(trainer.trainer.global_step)
    metrics = trainer.test(test_loader, **options)[0]
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('P2018 evaluation returned nonfinite metrics.')
    # Freeze constrained convolution weights for a stable exported inference state.
    model.requires_grad_(False).eval()
    probe = next(iter(test_loader))[0][:2]
    restored = EEGNet(**MODEL_CONFIG).requires_grad_(False).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(probe), model(probe), rtol=0, atol=0)
    preprocessing = dict(channels=CHANNELS, signal_unit='WFDB_header_native', sfreq=100, epoch_seconds=30,
                         l_freq=.5, h_freq=30., filter_design='Butterworth_8_SOS_zero_phase',
                         resampling='WFDB_per_channel_FFT_before_filter', channel_provenance=provenance,
                         normalization='per_epoch_per_channel', label_mapping=LABELS)
    partitions = dict(train_groups=sorted(str(value) for value in train.info.subject_id.unique()),
                      test_groups=sorted(str(value) for value in test.info.subject_id.unique()),
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
                  train_epochs=len(train), test_epochs=len(test), test_metrics=metrics,
                  strict_reload_equal=True, output_path=str(output),
                  scope='Released-record holdout within labeled training data; no challenge arousal or published benchmark score.')
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(test_group=str(groups[-1]), metrics=metrics, report=str(report_path))))


if __name__ == '__main__':
    main()
