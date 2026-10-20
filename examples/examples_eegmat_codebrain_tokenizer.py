"""
Fine-tune CodeBrain TFDual on EEGMat
====================================

Load the complete official CodeBrain_Tokenizer.pth, fine-tune both codebooks,
the encoder and both decoders with the six-term self-supervised objective,
and hold out one complete EEGMat subject. Cite Ma et al., CodeBrain, ICLR2026,
and Zyma et al. (2019), EEGMat, https://physionet.org/content/eegmat/1.0.0/.
CodeBrain code/weights use Apache-2.0; EEGMat uses ODC Attribution v1.0.
Use author checkpoint revision bef08d2fdb1759685371cc635aad21ce59163689 from
YjMajy/CodeBrain. No download or random/smaller fallback is performed.

Resample whole EEGMat records500->200Hz, apply MNE FIR0.3--75Hz/notch50Hz,
divide microvolts by100, and reshape four-second windows into [19,4,200].
Keep the exact source channel order/reference. Tokenizer positional rows1..19
are the author's sequential IDs, not a named montage. This teaching workflow
does not claim the pretraining channel mapping, record trimming or benchmark
protocol. Condition labels are recorded but are not training targets.

Train all but the last sorted complete subject for fixed epochs. Corrected
InfoNCE includes each positive once; the source duplicate-positive objective
is available through the model's explicit native compatibility option.
Test labels and reconstruction scores never select settings or epochs.
Disable usage-count tracking so held-out evaluation changes no model state;
training still updates codebooks by EMA and neural weights by Adam.

Save all model/EMA/BatchNorm buffers, tensor optimizer state with JSON groups,
Torch/global-loader RNG states, configuration/provenance JSON, held-out token
IDs and all three reconstructions. Strictly reload model and optimizer, then
require identical held-out predictions. JSON keeps floating configuration out
of old tensor-only unpicklers. Keep source records unchanged and use a fresh
cache if source or preprocessing changes. Results are functional checks,
not paper scores. The original checkpoint's known training-only metadata is
discarded safely; unknown pickle globals are rejected.

Run::

    python examples/examples_eegmat_codebrain_tokenizer.py --root-path ./eegmat \
        --checkpoint ./CodeBrain_Tokenizer.pth --epochs 1
"""
import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LIST
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import CodeBrainTokenizer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def filter_recording(eeg):
    eeg = mne.filter.filter_data(np.asarray(eeg, dtype=np.float64), sfreq=200,
                               l_freq=.3, h_freq=75., verbose=False)
    return mne.filter.notch_filter(eeg, Fs=200, freqs=[50.], verbose=False)


def prepare_window(eeg):
    if eeg.shape != (19, 800) or not np.isfinite(eeg).all():
        raise ValueError('Require19finite source channels and800samples at200Hz.')
    return (np.asarray(eeg, dtype=np.float64)/100.).astype(np.float32).reshape(19, 4, 200)


def standardize(values):
    return (values-values.mean(dim=(1, 2, 3), keepdim=True))/values.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)


@torch.no_grad()
def evaluate(model, dataset, batch_size):
    model.eval()
    outputs = {name: [] for name in ['time_ids', 'frequency_ids', 'raw', 'amplitude', 'phase', 'conditions']}
    totals = {name: 0. for name in ['time_quantization_loss', 'frequency_quantization_loss',
                                   'raw_mse', 'amplitude_mse', 'phase_mse']}
    loader = DataLoader(dataset, batch_size=batch_size, generator=torch.Generator().manual_seed(0))
    for eeg, target in loader:
        time, time_ids, time_loss, frequency, frequency_ids, frequency_loss, _ = model.encode(eeg, contrastive=False)
        raw = model.decode_t(time)
        amplitude, phase = model.decode_f(frequency)
        transformed = torch.fft.fft(eeg, dim=-1)
        values = dict(time_quantization_loss=time_loss, frequency_quantization_loss=frequency_loss,
            raw_mse=F.mse_loss(raw, eeg.flatten(1, 2)),
            amplitude_mse=F.mse_loss(amplitude, standardize(transformed.abs()).flatten(1, 2)),
            phase_mse=F.mse_loss(phase, standardize(transformed.angle()).flatten(1, 2)))
        for name, value in values.items():
            if not torch.isfinite(value):
                raise ValueError(f'Nonfinite held-out score: {name}.')
            totals[name] += value.item()*len(eeg)
        batch = dict(time_ids=time_ids.reshape(len(eeg), -1), frequency_ids=frequency_ids.reshape(len(eeg), -1),
                     raw=raw, amplitude=amplitude, phase=phase, conditions=target)
        for name, value in batch.items(): outputs[name].append(value.cpu())
    metrics = {name: value/len(dataset) for name, value in totals.items()}
    metrics['loss'] = sum(metrics.values())  # five eval terms; no contrastive term
    return {name: torch.cat(values) for name, values in outputs.items()}, metrics


def main():
    parser = argparse.ArgumentParser(description='Fine-tune complete pretrained CodeBrain TFDual on EEGMat.')
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache-path', default='./eegmat-codebrain-tokenizer-200hz-4s')
    parser.add_argument('--output-path', default='./eegmat-codebrain-tokenizer.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive.')
    torch.manual_seed(42)
    model = CodeBrainTokenizer.from_pretrained(args.checkpoint, track_usage=False, contrastive_mode='infonce')
    source = Path(args.root_path)
    source_hashes = {path.name: sha256(path) for path in sorted(source.glob('Subject*.edf'))}
    dataset = MentalArithmeticDataset(root_path=str(source), chunk_size=800, num_channel=19,
        sampling_rate=200, io_path=args.cache_path, io_mode='pickle', num_worker=0,
        before_trial=filter_recording,
        online_transform=transforms.Compose([transforms.Lambda(prepare_window), transforms.ToTensor()]),
        label_transform=transforms.Select('label'))
    subjects = sorted(dataset.info.subject_id.unique())
    if len(subjects) < 2:
        raise ValueError('At least two complete subject pairs are required.')
    for subject in subjects:
        if set(dataset.info.loc[dataset.info.subject_id == subject, 'label']) != {0, 1}:
            raise ValueError(f'Subject{subject} needs both recording conditions.')
    train, test = next(LeaveOneSubjectOut(split_path=args.cache_path+'-subject-split').split(dataset, subject=subjects[-1]))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=.0001)
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator)
    updates, train_losses = 0, []
    for _ in range(args.epochs):
        model.train(); total = 0.
        for eeg, _ in train_loader:
            optimizer.zero_grad()
            loss, _ = model(eeg)
            if not torch.isfinite(loss):
                raise ValueError('Training loss is not finite.')
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in parameters):
                raise ValueError('Training gradient is not finite.')
            optimizer.step(); updates += 1
            total += loss.item()*len(eeg)
        train_losses.append(total/len(train))
    predictions, metrics = evaluate(model, test, args.batch_size)
    config = dict(encoder_depth=model.encoder_depth, decoder_depth=model.decoder_depth, code_dim=model.code_dim,
        num_time_tokens=model.num_time_tokens, num_frequency_tokens=model.num_frequency_tokens,
        decay=model.decay, quantize_kmeans_init=model.quantize_kmeans_init, update_codebook=model.update_codebook,
        track_usage=model.track_usage, contrastive_mode=model.contrastive_mode)
    metadata = dict(checkpoint_sha256=sha256(args.checkpoint), source_sha256=source_hashes,
        channel_order=MENTAL_ARITHMETIC_CHANNEL_LIST, channel_ids=list(range(1, 20)),
        sampling_rate=200, window_samples=800, input_unit='microvolts divided by100',
        filters='Whole-record resample500->200Hz thenMNE FIR0.3--75Hz/notch50Hz',
        held_out_subject=subjects[-1], train_subjects=subjects[:-1], train_windows=len(train), test_windows=len(test),
        condition_labels_used_for_training=False, epochs=args.epochs, batch_size=args.batch_size,
        updates=updates, train_losses=train_losses, metrics=metrics, seed=42,
        optimizer=dict(name='Adam', lr=.0001), contrastive_mode='infonce', weights_license='Apache-2.0',
        torch_version=str(torch.__version__), usage_tracking=False, evaluation_contrastive=False)
    optimizer_state = optimizer.state_dict()
    artifact = dict(state_dict=model.state_dict(), model_config_json=json.dumps(config), metadata_json=json.dumps(metadata),
        optimizer_tensors=optimizer_state['state'], optimizer_groups_json=json.dumps(optimizer_state['param_groups']),
        torch_rng_state=torch.get_rng_state(), train_generator_state=generator.get_state(), predictions=predictions)
    output = Path(args.output_path); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    payload = torch.load(output, weights_only=True, map_location='cpu')
    restored = CodeBrainTokenizer(**json.loads(payload['model_config_json'])).eval()
    restored.load_state_dict(payload['state_dict'], strict=True)
    restored_optimizer = torch.optim.Adam([parameter for parameter in restored.parameters() if parameter.requires_grad], lr=.0001)
    restored_optimizer.load_state_dict(dict(state=payload['optimizer_tensors'], param_groups=json.loads(payload['optimizer_groups_json'])))
    restored_generator = torch.Generator().set_state(payload['train_generator_state'])
    if not torch.equal(restored_generator.get_state(), generator.get_state()):
        raise ValueError('Restored training-loader state differs.')
    for key, values in optimizer_state['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(restored_optimizer.state_dict()['state'][key][name], value, rtol=0, atol=0)
    torch.set_rng_state(payload['torch_rng_state'])
    restored_predictions, restored_metrics = evaluate(restored, test, args.batch_size)
    for name in predictions:
        torch.testing.assert_close(restored_predictions[name], predictions[name], rtol=0, atol=0)
    if restored_metrics != metrics:
        raise ValueError('Restored reconstruction scores differ.')
    if source_hashes != {path.name: sha256(path) for path in sorted(source.glob('Subject*.edf'))}:
        raise ValueError('Source recordings changed during the run.')
    print(json.dumps(dict(metadata, artifact=str(output), strict_restore=True)))


if __name__ == '__main__':
    main()
