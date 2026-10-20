"""
Probe pretrained TFM on EEGMat
===============================

Load the complete released CHB-MIT TFM tokenizer and encoder. Freeze both,
then train a new two-logit classifier on EEGMat background=0 versus mental
arithmetic=1. Cite Pradeepkumar et al., Tokenizing Single-Channel EEG with
Time-Frequency Motif Learning (ICLR2026), and Zyma et al. (2019),
https://physionet.org/content/eegmat/1.0.0/ (ODC Attribution v1.0).
Source and weights use MIT; retain LICENSES/TFMTokenizer.txt and the shared
linear-attention notices. Official source2d6da482b16dabbb2ebec808fa9f505fc6f367c4
contains both weights under pretrained_weigths/single_dataset_settings/
CHBMIT_tfm_tokenizer_2x2x8/. The tokenizer's complete released linear
reconstruction head is loaded explicitly, not omitted for a shape mismatch.
The encoder's original single-logit binary head is retained but unused by
the new two-class task. No automatic download or random fallback.

Resample complete records500->200Hz, filter MNE FIR0.1--75Hz/notch50Hz, and
split four-second windows. Select the FIRST16 EEGMat source electrodes,
explicitly excluding Fz/Cz/Pz. Normalize each window/channel by its95th
absolute percentile plus1e-8, without population-fitted statistics.
Magnitude STFT uses periodicHann200,hop100,centerFalse,bins0--99.
This yields seven IDs/channel; encoder position rows0--15 are sequential,
not a claimed physical mapping to the pretrained CHB-MIT bipolar montage.
This teaching transfer protocol does not reproduce CHB-MIT benchmark scores.

Hold out the last complete subject and train all others for fixed epochs.
Save all189tokenizer and59encoder tensors, the new head, model configuration,
source/checkpoint hashes and held-out predictions. Strictly restore and
require identical predictions; all pretrained state stays unchanged.
Use a fresh cache after changing preprocessing or sources.

Run::

    python examples/examples_eegmat_tfm.py --root-path ./eegmat \
        --tokenizer-checkpoint ./tfm_tokenizer_last.pth \
        --encoder-checkpoint ./tfm_encoder_best_model.pth --epochs 1
"""
import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LIST
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import TFMTokenizer, TFMEncoder, TFMClassifier


VARIANT = 'single-chbmit'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def filter_recording(eeg):
    eeg = mne.filter.filter_data(np.asarray(eeg, dtype=np.float64), sfreq=200,
        l_freq=.1, h_freq=75., verbose=False)
    return mne.filter.notch_filter(eeg, Fs=200, freqs=[50.], verbose=False)


def prepare_window(eeg):
    if eeg.shape != (19, 800) or not np.isfinite(eeg).all():
        raise ValueError('Require 19 finite source channels and 800 samples at200Hz.')
    eeg = np.asarray(eeg[:16], dtype=np.float64)
    scale = np.quantile(np.abs(eeg), .95, axis=-1, keepdims=True) + 1e-8
    return (eeg/scale).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Train a frozen complete pretrained CHB-MIT TFM EEGMat probe.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--tokenizer-checkpoint', required=True)
    parser.add_argument('--encoder-checkpoint', required=True)
    parser.add_argument('--cache-path', default='./eegmat-tfm-200hz-4s')
    parser.add_argument('--output-path', default='./eegmat-tfm.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive.')
    torch.manual_seed(42)
    tokenizer = TFMTokenizer.from_pretrained(args.tokenizer_checkpoint, update_codebook=False)
    encoder = TFMEncoder.from_pretrained(args.encoder_checkpoint)
    model = TFMClassifier(tokenizer, encoder, num_classes=2, freeze_encoder=True)
    source = Path(args.root_path)
    source_hashes = {path.name: sha256(path) for path in sorted(source.glob('Subject*.edf'))}
    dataset = MentalArithmeticDataset(root_path=str(source), chunk_size=800,
        num_channel=19, sampling_rate=200, io_path=args.cache_path, io_mode='pickle',
        num_worker=0, before_trial=filter_recording,
        online_transform=transforms.Compose([transforms.Lambda(prepare_window), transforms.ToTensor()]),
        label_transform=transforms.Select('label'))
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('At least two complete subject pairs are required.')
    for subject in subjects:
        if set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {0, 1}:
            raise ValueError(f'Subject {subject} needs both recording conditions.')
    train, test = next(LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split').split(
        dataset, subject=subjects[-1]))
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=.001)
    updates = 0
    for _ in range(args.epochs):
        model.train()
        for eeg, target in DataLoader(train, batch_size=args.batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(eeg), target)
            if not torch.isfinite(loss):
                raise ValueError('Training loss is not finite.')
            loss.backward()
            optimizer.step()
            updates += 1
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for eeg, target in DataLoader(test, batch_size=args.batch_size):
            logits.append(model(eeg).cpu())
            targets.append(target.cpu())
    logits, targets = torch.cat(logits), torch.cat(targets)
    metrics = dict(loss=nn.functional.cross_entropy(logits, targets).item(),
        accuracy=(logits.argmax(1) == targets).float().mean().item())
    tokenizer_config = dict(hid_channels=tokenizer.hid_channels, num_tokens=tokenizer.num_tokens,
        encoder_depth=tokenizer.encoder_depth, decoder_depth=tokenizer.decoder_depth,
        reconstruction_head=tokenizer.reconstruction_head, decay=tokenizer.decay,
        update_codebook=False, beta=tokenizer.beta)
    encoder_config = dict(hid_channels=encoder.hid_channels, num_tokens=encoder.num_tokens,
        num_layers=encoder.num_layers, num_classes=encoder.num_classes)
    metadata = dict(variant=VARIANT, checkpoint_sha256={'tokenizer': sha256(args.tokenizer_checkpoint), 'encoder': sha256(args.encoder_checkpoint)},
        source_sha256=source_hashes, channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST[:16], channel_ids=list(range(16)),
        sampling_rate=200, window_samples=800, input_unit='microvolts before normalization',
        normalization='per-window/channel abs95percentile+1e-8; no fitted population statistics',
        filters='MNE FIR0.1--75Hz and50Hznotch on complete recordings AFTER resampling to200Hz',
        label_mapping={'background': 0, 'mental_arithmetic': 1}, seed=42,
        held_out_subject=subjects[-1], train_subjects=subjects[:-1], epochs=args.epochs,
        train_windows=len(train), test_windows=len(test), updates=updates, metrics=metrics,
        optimizer={'name': 'Adam', 'lr': .001}, freeze_encoder=True,
        weights_license='MIT; retain tokenizer, encoder and shared linear-attention author notices')
    artifact = dict(state_dict=model.state_dict(), tokenizer_config_json=json.dumps(tokenizer_config),
        encoder_config_json=json.dumps(encoder_config),
        metadata_json=json.dumps(metadata), test_logits=logits, test_targets=targets)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    payload = torch.load(output, weights_only=True, map_location='cpu')
    restored = TFMClassifier(TFMTokenizer(**json.loads(payload['tokenizer_config_json'])),
        TFMEncoder(**json.loads(payload['encoder_config_json'])), num_classes=2, freeze_encoder=True).eval()
    restored.load_state_dict(payload['state_dict'], strict=True)
    restored_logits = []
    with torch.no_grad():
        for eeg, _ in DataLoader(test, batch_size=args.batch_size):
            restored_logits.append(restored(eeg))
    torch.testing.assert_close(torch.cat(restored_logits), logits, rtol=0, atol=0)
    for prefix, component in [('tokenizer', tokenizer), ('encoder', encoder)]:
        for name, value in component.state_dict().items():
            torch.testing.assert_close(payload['state_dict'][prefix + '.' + name], value, rtol=0, atol=0)
    if source_hashes != {path.name: sha256(path) for path in sorted(source.glob('Subject*.edf'))}:
        raise ValueError('Source files changed during the run.')
    print(json.dumps(dict(metadata, artifact=str(output), strict_restore=True)))


if __name__ == '__main__':
    main()
