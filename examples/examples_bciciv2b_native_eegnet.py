"""
Train native BCI IV 2b EEGNet
=============================

Read the labeled BNCI 004-2014 MAT release, B01T/B01E through B09T/B09E:
https://bnci-horizon-2020.eu/database/data-sets. Cite Leeb et al. and
Tangermann et al. The native MAT entry does not require MOABB or GDF files.

For one participant, train on the three released T sessions and evaluate the
two E sessions. These sessions belong to the same person. Select the three
bipolar EEG channels C3, Cz, C4; EOG is not a classification input. Use a
whole-session 4--38 Hz MNE FIR, resample to 128 Hz, then extract 3--7 seconds
relative to trial onset: one [1,3,512] input per trial. Reader values remain
in microvolts; normalize each window per channel, without fitting statistics
across participants or sessions. Add no notch or average reference.

Keep expert-flagged trials by default. ``--skip-artifacts`` excludes them
from both training and evaluation; record this different policy and select
a fresh cache. A trial-level accuracy/F1 result is not the original
competition's continuous, artifact-free time-resolved kappa score. This
example also differs from the separate MOABB participant-holdout workflow.

Install the source and run::

    pip install -e ".[dataset,trainer]"
    python examples/examples_bciciv2b_native_eegnet.py \
        --root-path ./BCICIV_2b_mat --subject 1 --epochs 1 \
        --output-path ./b01-native-eegnet.pt

The root may contain multiple participants; only the selected participant's
T/E rows enter the loaders. Every T and E session must remain nonempty.
Use fixed training epochs, without evaluation-based model selection. The
complete constructor, tensor state, preprocessing and actual phase/record
partition are exported safely with a JSON report. Existing outputs are
preserved. Use fresh caches after changing sources or preprocessing. The
gallery does not execute training; no published benchmark score is claimed.
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
from torcheeg.datasets import BCICIV2bDataset
from torcheeg.models import EEGNet
from torcheeg.trainers import ClassifierTrainer


MODEL_CONFIG = dict(num_electrodes=3, chunk_size=512, num_classes=2,
                    F1=8, F2=16, D=2, kernel_1=64, kernel_2=16, dropout=.25)
CHANNELS = ['C3', 'Cz', 'C4']
LABELS = {1: 0, 2: 1}


def main():
    parser = argparse.ArgumentParser(description='Train native BCI IV 2b EEGNet on T sessions and evaluate E sessions for one participant.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./bciciv2b-native-3ch-128hz-3to7')
    parser.add_argument('--output-path', default='./bciciv2b-native-eegnet.pt')
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
    dataset = BCICIV2bDataset(
        root_path=args.root_path, io_path=args.cache_path, io_mode='pickle', num_worker=0,
        sfreq=128, chunk_size=-1, overlap=0, num_channel=3,
        offset_start=750, offset_end=1750, l_freq=4., h_freq=38.,
        notch_freqs=None, ref_channels=None, skip_trial_with_artifacts=args.skip_artifacts,
        online_transform=transforms.Compose([
            transforms.MeanStdNormalize(axis=-1), transforms.ToTensor(), transforms.To2d()]),
        label_transform=transforms.Compose([transforms.Select('label'), transforms.Mapping(LABELS)]))
    train, test = copy.copy(dataset), copy.copy(dataset)
    selected = dataset.info['subject_id'] == args.subject
    train.info = dataset.info.loc[selected & (dataset.info['phase'] == 'train')].copy().reset_index(drop=True)
    test.info = dataset.info.loc[selected & (dataset.info['phase'] == 'test')].copy().reset_index(drop=True)
    if train.info.empty or test.info.empty:
        raise ValueError('The selected participant requires nonempty T training and E evaluation data.')
    if set(train.info['session_id']) != {1, 2, 3} or set(test.info['session_id']) != {4, 5}:
        raise ValueError('Require all three T sessions and both E sessions after the artifact policy.')
    if set(train.info['trial_id']) & set(test.info['trial_id']):
        raise ValueError('Training and evaluation trials must be disjoint.')
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=0)
    model = EEGNet(**MODEL_CONFIG)
    trainer = ClassifierTrainer(model=model, num_classes=2, lr=.001, accelerator='cpu',
                                devices=1, metrics=['accuracy', 'f1score'])
    options = dict(logger=False, enable_checkpointing=False, enable_progress_bar=False,
                   enable_model_summary=False)
    trainer.fit(train_loader, None, max_epochs=args.epochs, num_sanity_val_steps=0,
                default_root_dir=str(output.parent / 'bciciv2b-native-training'), **options)
    steps = int(trainer.trainer.global_step)
    metrics = trainer.test(test_loader, **options)[0]
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('BCI IV 2b evaluation returned nonfinite metrics.')
    # Freeze constrained convolution weights for a stable exported inference state.
    model.requires_grad_(False).eval()
    probe = next(iter(test_loader))[0][:2]
    restored = EEGNet(**MODEL_CONFIG).requires_grad_(False).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(probe), model(probe), rtol=0, atol=0)
    preprocessing = dict(channels=CHANNELS, signal_unit='uV', sfreq=128,
                         source_offset_start=750, source_offset_end=1750, source_sfreq=250,
                         chunk_size=-1, overlap=0, num_channel=3,
                         l_freq=4., h_freq=38., notch_freqs=None, ref_channels=None,
                         processing='continuous_session_MNE_FIR_then_resample',
                         normalization='per_window_per_channel', source_label_mapping=LABELS,
                         skip_trial_with_artifacts=args.skip_artifacts)
    partitions = dict(subject=args.subject, train_phase='train', test_phase='test',
                      train_sessions=sorted(int(value) for value in train.info.session_id.unique()),
                      test_sessions=sorted(int(value) for value in test.info.session_id.unique()),
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
                  scope='One-participant released T/E phase holdout; no continuous competition kappa or published benchmark claim.')
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(subject=args.subject, metrics=metrics, report=str(report_path))))


if __name__ == '__main__':
    main()
