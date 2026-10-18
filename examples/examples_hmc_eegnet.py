"""
Train EEGNet on HMC
========================

Use the Haaglanden Medisch Centrum sleep staging database:
https://physionet.org/content/hmc-sleep-staging/1.1/.
Cite Alvarez-Estevez and Rijsman (2021) and PhysioNet. Keep the released
SN EDF/scoring pairs in the recordings directory.

Select EEG C4-M1 and EOG E1-M2 in that order, apply a 0.5--30 Hz FIR filter,
resample to 100 Hz and read complete 30-second scored epochs in volts.
Normalize each epoch per channel. Keep the final released SN group in
sorted order for testing, and train for a fixed epoch count. The public
SN group identifies a released night; it does not establish person linkage
between different nights. Test data are not used for parameter selection.

Install the source and run::

    pip install -e ".[dataset,trainer]"
    python examples/examples_hmc_eegnet.py --root-path ./HMC/recordings \
        --epochs 1 --output-path ./hmc-eegnet.pt

At least two complete SN groups are required. The default reads each whole
scored night. Use ``--start-at`` and ``--end-at`` for an explicit crop, and
a new cache path after changing source data, channels, rate, filters or crop.
Cropped checks do not establish whole-night five-class performance. This
standalone example does not reproduce a published sleep-staging benchmark.
The documentation gallery does not execute training.
"""

import argparse
import json
import math
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import HMCDataset
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import EEGNet
from torcheeg.trainers import ClassifierTrainer


MODEL_CONFIG = dict(num_electrodes=2, chunk_size=3000, num_classes=5,
                    F1=8, F2=16, D=2, kernel_1=64, kernel_2=16, dropout=.25)
CHANNELS = ['EEG C4-M1', 'EOG E1-M2']
LABELS = {'Sleep stage W': 0, 'Sleep stage N1': 1, 'Sleep stage N2': 2,
          'Sleep stage N3': 3, 'Sleep stage R': 4}


def main():
    parser = argparse.ArgumentParser(description='Train EEGNet on HMC with one held-out released SN group.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./hmc-eeg-eog-100hz-30s')
    parser.add_argument('--output-path', default='./hmc-eegnet.pt')
    parser.add_argument('--start-at', type=float, default=0.)
    parser.add_argument('--end-at', type=float, default=None)
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
    dataset = HMCDataset(
        root_path=args.root_path, io_path=args.cache_path, io_mode='pickle', num_worker=0,
        channels=CHANNELS, sfreq=100, l_freq=.5, h_freq=30.,
        start_at=args.start_at, end_at=args.end_at, missing_channels='raise',
        online_transform=transforms.Compose([
            transforms.MeanStdNormalize(axis=-1), transforms.ToTensor(), transforms.To2d()]),
        label_transform=transforms.Compose([transforms.Select('label'), transforms.Mapping(LABELS)]))
    groups = sorted(dataset.info['subject_id'].unique(), key=str)
    if len(groups) < 2:
        raise ValueError('At least two complete released SN groups are required.')
    splitter = LeaveOneSubjectOut(split_path=args.cache_path + '-sn-split')
    train, test = next(splitter.split(dataset, subject=groups[-1]))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=0)
    model = EEGNet(**MODEL_CONFIG)
    trainer = ClassifierTrainer(model=model, num_classes=5, lr=.001, accelerator='cpu',
                                devices=1, metrics=['accuracy', 'f1score'])
    options = dict(logger=False, enable_checkpointing=False, enable_progress_bar=False,
                   enable_model_summary=False)
    trainer.fit(train_loader, None, max_epochs=args.epochs, num_sanity_val_steps=0,
                default_root_dir=str(output.parent / 'hmc-training'), **options)
    steps = int(trainer.trainer.global_step)
    metrics = trainer.test(test_loader, **options)[0]
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('HMC evaluation returned nonfinite metrics.')
    # Freeze constrained convolution weights for a stable exported inference state.
    model.requires_grad_(False).eval()
    probe = next(iter(test_loader))[0][:2]
    restored = EEGNet(**MODEL_CONFIG).requires_grad_(False).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(probe), model(probe), rtol=0, atol=0)
    preprocessing = dict(channels=CHANNELS, signal_unit='V', sfreq=100, epoch_seconds=30,
                         l_freq=.5, h_freq=30., start_at=args.start_at, end_at=args.end_at,
                         normalization='per_epoch_per_channel', label_mapping=LABELS)
    partitions = dict(train_groups=sorted(str(value) for value in train.info.subject_id.unique()),
                      test_groups=sorted(str(value) for value in test.info.subject_id.unique()))
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
                  scope='Released-SN holdout; no cross-night person linkage or published benchmark claim.')
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(test_group=str(groups[-1]), metrics=metrics, report=str(report_path))))


if __name__ == '__main__':
    main()
