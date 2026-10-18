"""
Train native BCI IV 2a EEGNet
=============================

Read the labeled BNCI 001-2014 MAT release, A01T/A01E through A09T/A09E:
https://bnci-horizon-2020.eu/database/data-sets. Cite Brunner et al. and
Tangermann et al. This native reader does not require MOABB or GDF files.

For one participant, train on the T recording and evaluate the E recording.
Both recordings belong to the same person and each has six imagery runs.
Select the first 22 EEG channels; EOG is not a classification input. Use a
whole-run 4--38 Hz MNE FIR, resample to 128 Hz, then extract 2--6 seconds
relative to trial onset: one [1,22,512] input per trial. The cue starts at
two seconds and imagery ends at six seconds. Native values remain in
microvolts; normalize each window per channel. Add no notch or average
reference and fit no normalization statistics across trials or recordings.

Keep expert-flagged trials by default. ``--skip-artifacts`` excludes them
from both training and evaluation; use a fresh cache for this policy.
This fixed-epoch, offline trial-level evaluation is not the original
competition's causal continuous-output, artifact-free kappa evaluation.

Install the source and run::

    pip install -e ".[dataset,trainer]"
    python examples/examples_bciciv2a_native_eegnet.py \
        --root-path ./BCICIV_2a_mat --subject 1 --epochs 1 \
        --output-path ./a01-native-eegnet.pt

The root may contain multiple participants; only the selected participant's
T/E rows enter the loaders. Each of its six T and six E runs must remain
nonempty. Never select models or epochs with E labels. The complete EEGNet
constructor, tensor state, preprocessing and actual record/run partitions
are exported safely with a JSON report. Existing outputs are preserved.
The documentation gallery does not execute training. No published score is
claimed; long experiments must state all preprocessing and evaluation choices.
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
from torcheeg.datasets import BCICIV2aDataset
from torcheeg.models import EEGNet
from torcheeg.trainers import ClassifierTrainer


MODEL_CONFIG = dict(num_electrodes=22, chunk_size=512, num_classes=4,
                    F1=8, F2=16, D=2, kernel_1=64, kernel_2=16, dropout=.25)
CHANNELS = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
            'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
LABELS = {1: 0, 2: 1, 3: 2, 4: 3}


def main():
    parser = argparse.ArgumentParser(description='Train native BCI IV 2a EEGNet on the T recording and evaluate the E recording for one participant.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./bciciv2a-native-22ch-128hz-2to6')
    parser.add_argument('--output-path', default='./bciciv2a-native-eegnet.pt')
    parser.add_argument('--subject', type=int, choices=range(1, 10), default=1)
    parser.add_argument('--skip-artifacts', action='store_true')
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
    dataset = BCICIV2aDataset(
        root_path=args.root_path, io_path=args.cache_path, io_mode='pickle', num_worker=0,
        sfreq=128, chunk_size=-1, overlap=0, num_channel=22,
        offset_start=500, offset_end=1500, l_freq=4., h_freq=38.,
        notch_freqs=None, ref_channels=None, skip_trial_with_artifacts=args.skip_artifacts,
        online_transform=transforms.Compose([
            transforms.MeanStdNormalize(axis=-1), transforms.ToTensor(), transforms.To2d()]),
        label_transform=transforms.Compose([transforms.Select('label'), transforms.Mapping(LABELS)]))
    train, test = copy.copy(dataset), copy.copy(dataset)
    subject_id = f'A{args.subject:02d}'
    selected = dataset.info['subject_id'] == subject_id
    train.info = dataset.info.loc[selected & (dataset.info['phase'] == 'train')].copy().reset_index(drop=True)
    test.info = dataset.info.loc[selected & (dataset.info['phase'] == 'test')].copy().reset_index(drop=True)
    if train.info.empty or test.info.empty:
        raise ValueError('The selected participant requires nonempty T training and E evaluation data.')
    if train.info['run'].nunique() != 6 or test.info['run'].nunique() != 6:
        raise ValueError('Require all six T runs and all six E runs after the artifact policy.')
    if set(train.info['trial_id']) & set(test.info['trial_id']):
        raise ValueError('Training and evaluation trials must be disjoint.')
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=0)
    model = EEGNet(**MODEL_CONFIG)
    trainer = ClassifierTrainer(model=model, num_classes=4, lr=.001, accelerator='cpu',
                                devices=1, metrics=['accuracy', 'f1score'])
    options = dict(logger=False, enable_checkpointing=False, enable_progress_bar=False,
                   enable_model_summary=False)
    trainer.fit(train_loader, None, max_epochs=args.epochs, num_sanity_val_steps=0,
                default_root_dir=str(output.parent / 'bciciv2a-native-training'), **options)
    steps = int(trainer.trainer.global_step)
    metrics = trainer.test(test_loader, **options)[0]
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('BCI IV 2a evaluation returned nonfinite metrics.')
    # Freeze constrained convolution weights for a stable exported inference state.
    model.requires_grad_(False).eval()
    probe = next(iter(test_loader))[0][:2]
    restored = EEGNet(**MODEL_CONFIG).requires_grad_(False).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(probe), model(probe), rtol=0, atol=0)
    preprocessing = dict(channels=CHANNELS, signal_unit='uV', sfreq=128,
                         source_offset_start=500, source_offset_end=1500, source_sfreq=250,
                         chunk_size=-1, overlap=0, num_channel=22,
                         l_freq=4., h_freq=38., notch_freqs=None, ref_channels=None,
                         processing='continuous_run_MNE_FIR_then_resample',
                         normalization='per_window_per_channel', source_label_mapping=LABELS,
                         skip_trial_with_artifacts=args.skip_artifacts)
    partitions = dict(subject_id=subject_id, train_phase='train', test_phase='test',
                      train_sessions=sorted(str(value) for value in train.info.session.unique()),
                      test_sessions=sorted(str(value) for value in test.info.session.unique()),
                      train_runs=sorted(int(value) for value in train.info.run.unique()),
                      test_runs=sorted(int(value) for value in test.info.run.unique()),
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
                  train_trials=len(train), test_trials=len(test),
                  train_flagged_trials=int(train.info.artifact.sum()),
                  test_flagged_trials=int(test.info.artifact.sum()), test_metrics=metrics,
                  strict_reload_equal=True, output_path=str(output),
                  scope='One-participant released T/E recording holdout; no causal continuous competition kappa or published benchmark claim.')
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(subject=args.subject, metrics=metrics, report=str(report_path))))


if __name__ == '__main__':
    main()
