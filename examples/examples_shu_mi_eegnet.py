"""
Train EEGNet on SHU motor imagery
=================================

Use Ma et al. (2022), SHU Multi-session Dataset, trial MAT files:
https://figshare.com/articles/code/shu_dataset/19228725/1.
Cite the paper and the dataset. Each MAT file is one participant/session,
with retained four-second trials at 250 Hz in microvolts. The provider has
already removed bad segments and baselines and applied 0.5--40 Hz FIR
filtering. This example adds no filter, channel repair or amplitude scaling.

Keep all 32 published channels, resample each full trial to 200 Hz with a
float64 SciPy FFT and normalize each trial per channel. Each model input is
[1,32,800]. Do not split a four-second trial into independently assigned
windows. Hold out the final sorted participant ID, keeping all their sessions
in the test set, and train for a fixed epoch count without test-based model
selection. This is an explicit cross-participant workflow, not the paper's
within-session, cross-session or adaptation benchmark.

Install the source and run::

    pip install -e ".[dataset,trainer]"
    python examples/examples_shu_mi_eegnet.py --root-path ./SHU-MI \
        --epochs 1 --output-path ./shu-mi-eegnet.pt

At least two participants are required. All selected MAT trials are used,
including sessions with fewer than 100 retained trials. Use fresh caches
after changing the source version, membership or preprocessing. The safe
export stores the complete constructor, tensor state, preprocessing and
participant/session record IDs, with a JSON evaluation report. Existing
output paths are preserved. The gallery does not execute training.
"""

import argparse
import json
import math
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import SHUMIDataset
from torcheeg.datasets.constants import SHU_MI_CHANNEL_LIST
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import EEGNet
from torcheeg.trainers import ClassifierTrainer


MODEL_CONFIG = dict(num_electrodes=32, chunk_size=800, num_classes=2,
                    F1=8, F2=16, D=2, kernel_1=64, kernel_2=16, dropout=.25)
CHANNELS = SHU_MI_CHANNEL_LIST
LABELS = {1: 0, 2: 1}


def main():
    parser = argparse.ArgumentParser(description='Train EEGNet on SHU-MI with one held-out participant and all their sessions.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--cache-path', default='./shu-mi-full-trial-200hz')
    parser.add_argument('--output-path', default='./shu-mi-eegnet.pt')
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
    dataset = SHUMIDataset(
        root_path=args.root_path, io_path=args.cache_path, io_mode='pickle', num_worker=0,
        sfreq=200, chunk_size=-1, overlap=0, num_channel=32,
        online_transform=transforms.Compose([
            transforms.MeanStdNormalize(axis=-1), transforms.ToTensor(), transforms.To2d()]),
        label_transform=transforms.Select('label'))
    groups = sorted(dataset.info['subject_id'].unique(), key=str)
    if len(groups) < 2:
        raise ValueError('At least two participants with complete trials are required.')
    splitter = LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split')
    train, test = next(splitter.split(dataset, subject=groups[-1]))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=0)
    model = EEGNet(**MODEL_CONFIG)
    trainer = ClassifierTrainer(model=model, num_classes=2, lr=.001, accelerator='cpu',
                                devices=1, metrics=['accuracy', 'f1score'])
    options = dict(logger=False, enable_checkpointing=False, enable_progress_bar=False,
                   enable_model_summary=False)
    trainer.fit(train_loader, None, max_epochs=args.epochs, num_sanity_val_steps=0,
                default_root_dir=str(output.parent / 'shu-mi-training'), **options)
    steps = int(trainer.trainer.global_step)
    metrics = trainer.test(test_loader, **options)[0]
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('SHU-MI evaluation returned nonfinite metrics.')
    # Freeze constrained convolution weights for a stable exported inference state.
    model.requires_grad_(False).eval()
    probe = next(iter(test_loader))[0][:2]
    restored = EEGNet(**MODEL_CONFIG).requires_grad_(False).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(probe), model(probe), rtol=0, atol=0)
    preprocessing = dict(channels=CHANNELS, signal_unit='uV', sfreq=200, trial_seconds=4,
                         chunk_size=-1, overlap=0, num_channel=32,
                         resampling='SciPy_float64_FFT', additional_filter=None,
                         provider_preprocessing='bad_segment_and_baseline_removal_FIR_0.5_40Hz',
                         normalization='per_trial_per_channel', source_label_mapping=LABELS)
    partitions = dict(train_subjects=sorted(str(value) for value in train.info.subject_id.unique()),
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
                  train_trials=len(train), test_trials=len(test), test_metrics=metrics,
                  strict_reload_equal=True, output_path=str(output),
                  scope='Participant holdout with all sessions together; no published benchmark claim.')
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(test_group=str(groups[-1]), metrics=metrics, report=str(report_path))))


if __name__ == '__main__':
    main()
