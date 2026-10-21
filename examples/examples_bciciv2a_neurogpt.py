"""
Probe complete external NeuroGPT on BCI-IV 2a
=============================================

Explicitly supply the trusted wenhui0206/NeuroGPT source at
230571d45ca4369b82f33ab42fc00863c2f95598 and the official full Hugging Face
wenhuic/Neuro-GPT checkpoint. Both are separately supplied GPL-3.0 materials;
no author implementation code or pretrained weights are distributed here.
Retain all 193 states: native Conformer, embedder, complete causal GPT,
unembedder and release CLS/vocabulary/pooler states.

Use native 250 Hz data, all 22 physical EEG channels in provider order and
its original left mastoid reference. Explicitly acknowledge transfer from
the different original TUH pretraining montage; no electrode is renamed. Exclude EOG and marked artifact trials.
Filter whole EEG runs with MNE FIR 0.5--40 Hz. Extract exactly four seconds
at trial onset + two seconds (cue onset); the reader converts MATLAB onsets
to Python indices before the offset. Normalize each trial/channel using
population z-scores and a 1e-6 microvolt standard-deviation floor. Reshape
into two nonoverlapping 500-sample segments, without invented electrodes.

Hold out the last complete subject and train only a new linear head on the
last unmasked native GPT feature. This is an explicit short-context teaching
transfer protocol; original pretraining uses 32 segments with 50-sample
overlap, and the author's downstream decoding head differs. No published
benchmark score or original record-level normalization is reproduced here.
Export and strictly restore every model state, physical metadata, source
and checkpoint hashes and held-out predictions.

BCI-IV 2a/BNCI 001-2014 MATLAB files: obtain A01T.mat, A02T.mat, etc. from
https://bnci-horizon-2020.eu/database/data-sets (CC BY-ND 4.0). Cite
Tangermann et al. (2012), Review of the BCI competition IV, and the NeuroGPT
paper https://arxiv.org/abs/2311.03764. Data/weights are never bundled.
The explicit author backend needs transformers and torchinfo separately.

Run::

    python examples/examples_bciciv2a_neurogpt.py --root-path ./BCICIV_2a_mat \
        --backend-root ./NeuroGPT --checkpoint ./pytorch_model.bin --epochs 1
"""

import argparse
import hashlib
import importlib.util
from importlib.machinery import ModuleSpec
import json
import subprocess
import sys
import types
from pathlib import Path

import mne
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import BCICIV2aDataset
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import NeuroGPT, NeuroGPTClassifier

VARIANT = 'complete-causal-external-backend'
SOURCE_REVISION = '230571d45ca4369b82f33ab42fc00863c2f95598'
CHANNELS = [
    'Fz',
    'FC3',
    'FC1',
    'FCz',
    'FC2',
    'FC4',
    'C5',
    'C3',
    'C1',
    'Cz',
    'C2',
    'C4',
    'C6',
    'CP3',
    'CP1',
    'CPz',
    'CP2',
    'CP4',
    'P1',
    'Pz',
    'P2',
    'POz',
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_neurogpt_backend(root, checkpoint_path, channel_names):
    """Explicitly construct every original component from trusted local code."""
    root = Path(root).resolve()
    revision = subprocess.run(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError('Use the documented pinned complete NeuroGPT author revision.')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)
    # Original source uses bare package imports. Keep those exact original
    # model/helper bodies while refusing to replace another loaded package.
    for name in ['encoder', 'embedder', 'decoder']:
        folder = root / 'src' / name
        existing = sys.modules.get(name)
        if existing is not None:
            if [Path(v).resolve() for v in getattr(existing, '__path__', [])] != [
                folder
            ]:
                raise ValueError(
                    'Use a fresh process; another author package name is already loaded.'
                )
        else:
            package = types.ModuleType(name)
            package.__path__ = [str(folder)]
            package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
            package.__spec__.submodule_search_locations = [str(folder)]
            sys.modules[name] = package
    from encoder.conformer_braindecode import EEGConformer
    from embedder.make import make_embedder
    from decoder.make_decoder import make_decoder
    from decoder.unembedder import make_unembedder

    name = (
        '_torcheeg_external_neurogpt_'
        + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    )
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / 'src/model.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    encoder = EEGConformer(
        n_outputs=4, n_chans=22, n_times=500, ch_pos=None, is_decoding_mode=False
    )
    embedder = make_embedder(
        architecture='GPT',
        training_style='CSM_causal',
        in_dim=1080,
        embed_dim=1024,
        num_hidden_layers=1,
        dropout=0.1,
        n_positions=512,
    )
    decoder = make_decoder(
        architecture='GPT',
        num_hidden_layers=6,
        embed_dim=1024,
        num_attention_heads=16,
        n_positions=512,
        intermediate_dim_factor=4,
        hidden_activation='gelu_new',
        dropout=0.1,
    )
    # Retain the original eager attention protocol with newer Transformers.
    decoder.transformer.config._attn_implementation = 'eager'
    if hasattr(decoder.transformer, '_attn_implementation'):
        decoder.transformer._attn_implementation = 'eager'
    unembedder = make_unembedder(
        embed_dim=1024, num_hidden_layers=1, out_dim=1080, dropout=0.1
    )
    backend = sys.modules[name].Model(encoder, embedder, decoder, unembedder)
    return backend, dict(
        conformer_layers=6,
        gpt_layers=6,
        gpt_heads=16,
        gpt_width=1024,
        segment_samples=500,
        encoder_feature_size=1080,
        attention='eager',
        transformers_version=__import__('transformers').__version__,
    )


def filter_recording(eeg):
    values = np.asarray(eeg, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[0] < 22 or not np.isfinite(values[:22]).all():
        raise ValueError('Require finite source EEG runs with at least 22 channels.')
    values[:22] = mne.filter.filter_data(
        values[:22], sfreq=250, l_freq=0.5, h_freq=40.0, verbose=False
    )
    return values


def prepare_window(eeg):
    values = np.asarray(eeg, dtype=np.float64)
    if values.shape != (22, 1000) or not np.isfinite(values).all():
        raise ValueError('Require finite cue-aligned EEG [22,1000] at 250 Hz.')
    values = (
        (values - values.mean(-1, keepdims=True))
        / values.std(-1, keepdims=True).clip(min=1e-6)
    ).astype(np.float32)
    return values.reshape(22, 2, 500).transpose(1, 0, 2).copy()


def main():
    parser = argparse.ArgumentParser(
        description='Train a complete frozen NeuroGPT BCI-IV 2a probe.'
    )
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--backend-root', required=True)
    parser.add_argument('--cache-path', default='./bciciv2a-neurogpt-cue-4s')
    parser.add_argument('--output-path', default='./bciciv2a-neurogpt.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive.')
    torch.manual_seed(42)
    backend, backend_config = build_neurogpt_backend(
        args.backend_root, args.checkpoint, CHANNELS
    )
    backbone = NeuroGPT.from_pretrained(
        args.checkpoint, backend, CHANNELS, allow_montage_transfer=True
    )
    model = NeuroGPTClassifier(backbone, num_classes=4, freeze_backbone=True)
    source = Path(args.root_path)
    source_hashes = {path.name: sha256(path) for path in sorted(source.glob('*.mat'))}
    dataset = BCICIV2aDataset(
        root_path=str(source),
        offset=500,
        chunk_size=1000,
        num_channel=22,
        skip_trial_with_artifacts=True,
        io_path=args.cache_path,
        io_mode='pickle',
        num_worker=0,
        before_trial=filter_recording,
        online_transform=transforms.Compose(
            [transforms.Lambda(prepare_window), transforms.ToTensor()]
        ),
        label_transform=transforms.Compose(
            [transforms.Select('label'), transforms.Lambda(lambda value: value - 1)]
        ),
    )
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('At least two distinct complete subjects are required.')
    for subject in subjects:
        if set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {
            1,
            2,
            3,
            4,
        }:
            raise ValueError(f'Subject {subject} needs all four motor imagery classes.')
    train, test = next(
        LeaveOneSubjectOut(split_path=args.cache_path + '-subject-split').split(
            dataset, subject=subjects[-1]
        )
    )
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
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
    metrics = dict(
        loss=nn.functional.cross_entropy(logits, targets).item(),
        accuracy=(logits.argmax(1) == targets).float().mean().item(),
    )
    config = dict(
        channel_names=list(backbone.channel_names),
        sampling_rate=backbone.sampling_rate,
        allow_montage_transfer=backbone.allow_montage_transfer,
    )
    metadata = dict(
        variant=VARIANT,
        checkpoint_sha256=sha256(args.checkpoint),
        source_sha256=source_hashes,
        channel_order=CHANNELS,
        allow_montage_transfer=True,
        pretraining_channel_order=list(NeuroGPT.PRETRAINING_CHANNELS),
        reference='source left mastoid reference preserved; no re-referencing applied',
        sampling_rate=250,
        window_samples=1000,
        cue_offset_samples=500,
        segment_samples=500,
        segment_overlap=0,
        segment_count=2,
        skip_trial_with_artifacts=True,
        input_unit='microvolts before normalization',
        normalization='per-window/channel population z-score with std floor1e-6',
        filters='MNE FIR 0.5--40 Hz on complete EEG runs at native 250 Hz; original acquisition filtering/reference preserved',
        label_mapping={'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3},
        seed=42,
        held_out_subject=subjects[-1],
        train_subjects=subjects[:-1],
        epochs=args.epochs,
        train_windows=len(train),
        test_windows=len(test),
        updates=updates,
        metrics=metrics,
        optimizer={'name': 'Adam', 'lr': 0.001},
        freeze_backbone=True,
        weights_license='GPL-3.0 author source and weights obtained separately; not bundled',
        backend_revision=SOURCE_REVISION,
        backend_config=backend_config,
        mne_version=str(mne.__version__),
        features='last unmasked native causal GPT segment feature; new 4-class task head',
    )
    artifact = dict(
        state_dict=model.state_dict(),
        backbone_config_json=json.dumps(config),
        metadata_json=json.dumps(metadata),
        test_logits=logits,
        test_targets=targets,
    )
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    payload = torch.load(output, weights_only=True, map_location='cpu')
    restored_config = json.loads(payload['backbone_config_json'])
    restored_backend, _ = build_neurogpt_backend(
        args.backend_root, args.checkpoint, restored_config['channel_names']
    )
    restored = NeuroGPTClassifier(
        NeuroGPT(restored_backend, **restored_config),
        num_classes=4,
        freeze_backbone=True,
    ).eval()
    restored.load_state_dict(payload['state_dict'], strict=True)
    restored_logits = []
    with torch.no_grad():
        for eeg, _ in DataLoader(test, batch_size=args.batch_size):
            restored_logits.append(restored(eeg))
    torch.testing.assert_close(torch.cat(restored_logits), logits, rtol=0, atol=0)
    for name, value in backbone.state_dict().items():
        torch.testing.assert_close(
            payload['state_dict']['backbone.' + name], value, rtol=0, atol=0
        )
    if source_hashes != {
        path.name: sha256(path) for path in sorted(source.glob('*.mat'))
    }:
        raise ValueError('Source files changed during the run.')
    print(json.dumps(dict(metadata, artifact=str(output), strict_restore=True)))


if __name__ == '__main__':
    main()
