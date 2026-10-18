import csv
import json
import re
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, Tuple

import mne
import numpy as np
from scipy.io import loadmat

from ..base_dataset import BaseDataset
from ....utils import get_random_dir_path


CHANNELS = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']
LABELS = {'C': 0, 'A': 1, 'F': 2}
GROUP_NAMES = {'C': 'HC', 'A': 'AD', 'F': 'FTD'}
SOURCE_SFREQ = 500


def _settings(chunk_size, overlap, num_channel, sfreq,
              notch_freqs=None, l_freq=None, h_freq=None, ref_channels=None):
    for value in (chunk_size, overlap, num_channel, sfreq):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError('Chunk, overlap, channel and sampling settings must be integers.')
    if (not 1 <= num_channel <= 19 or sfreq <= 0 or overlap < 0
            or (chunk_size > 0 and overlap >= chunk_size)):
        raise ValueError('Require 1--19 channels, a positive rate and overlap smaller than a positive chunk.')
    for value in (notch_freqs, l_freq, h_freq):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or not 0 < value < SOURCE_SFREQ / 2):
            raise ValueError('Filter frequencies must be finite, positive and below source Nyquist.')
    if l_freq is not None and h_freq is not None and l_freq >= h_freq:
        raise ValueError('l_freq must be smaller than h_freq.')
    if ref_channels not in (None, 'average'):
        raise ValueError('ref_channels must be None or average.')


def _participants(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file, delimiter='\t')
        reader.fieldnames = [name.strip().lstrip('\ufeff') for name in (reader.fieldnames or [])]
        if (len(set(reader.fieldnames)) != len(reader.fieldnames)
                or not {'participant_id', 'Group'} <= set(reader.fieldnames)):
            raise ValueError('participants.tsv needs unique participant_id and Group columns.')
        people = {}; numeric_ids = set()
        for row in reader:
            subject, group = (row.get('participant_id') or '').strip(), (row.get('Group') or '').strip()
            if not re.fullmatch(r'sub-[0-9]+', subject) or int(subject[4:]) < 1:
                raise ValueError('ADFTD participant IDs must be positive numeric BIDS IDs.')
            if subject in people or int(subject[4:]) in numeric_ids or group not in LABELS:
                raise ValueError('ADFTD participant IDs must be unique and diagnosis must be C, A or F.')
            people[subject] = group; numeric_ids.add(int(subject[4:]))
    if not people:
        raise ValueError('No ADFTD participants are listed.')
    return people


def _load_set(path):
    content = loadmat(path, simplify_cells=True)
    header = content.get('EEG', content)
    dimensions = {}
    for key in ('nbchan', 'srate', 'pnts', 'trials'):
        value = np.asarray(header[key])
        if value.size != 1 or value.dtype.kind not in 'iuf' or not np.isfinite(value).all():
            raise ValueError(f'EEGLAB {key} must be a finite scalar.')
        value = value.item()
        if value <= 0 or value != int(value):
            raise ValueError(f'EEGLAB {key} must be a positive integer.')
        dimensions[key] = int(value)
    if dimensions['nbchan'] != 19 or dimensions['srate'] != 500 or dimensions['trials'] != 1:
        raise ValueError('Require released continuous 19-channel, 500 Hz ADFTD SET records.')
    labels = [str(channel['labels']).upper() for channel in header['chanlocs']]
    if len(labels) != 19 or set(labels) != set(CHANNELS):
        raise ValueError('SET labels must contain each canonical ADFTD channel once.')
    values = header['data']
    if isinstance(values, str):
        expected = Path(path).with_suffix('.fdt')
        if values != expected.name or expected.stat().st_size != 19 * dimensions['pnts'] * 4:
            raise ValueError('External SET data must name a matching local FDT with the declared byte count.')
        values = np.fromfile(expected, dtype='<f4').reshape(-1, 19).T
    values = np.asarray(values)
    if (values.shape != (19, dimensions['pnts']) or values.dtype.kind not in 'iuf'
            or not np.isfinite(values).all()):
        raise ValueError('SET data must match the declared finite channel/time geometry.')
    xmin = np.asarray(header.get('xmin', 0.))
    if xmin.size != 1 or xmin.dtype.kind not in 'iuf' or not np.isfinite(xmin).all() or xmin.item() != 0:
        raise ValueError('The released continuous SET clock must start at zero.')
    xmax = np.asarray(header.get('xmax', (dimensions['pnts'] - 1) / 500))
    if (xmax.size != 1 or xmax.dtype.kind not in 'iuf' or not np.isfinite(xmax).all()
            or abs(xmax.item() - (dimensions['pnts'] - 1) / 500) > 1 / 500):
        raise ValueError('SET xmax must agree with its inclusive last sample.')
    events = header.get('event', [])
    events = [events] if isinstance(events, dict) else list(events)
    cuts = {0, dimensions['pnts']}
    for event in events:
        if str(event.get('type', '')).strip().lower() != 'boundary':
            continue
        latency = np.asarray(event['latency'])
        if latency.size != 1 or latency.dtype.kind not in 'iuf' or not np.isfinite(latency).all():
            raise ValueError('EEGLAB boundary latency must be finite.')
        latency = float(latency.item())
        if not .5 <= latency <= dimensions['pnts'] + .5:
            raise ValueError('EEGLAB boundary latency is outside the stored recording.')
        # A latency n + 0.5 marks the cut after the nth one-based sample.
        cuts.add(int(np.floor(latency)))
    order = [labels.index(channel) for channel in CHANNELS]
    return values[order].astype(np.float64) * 1e-6, sorted(cuts)


class _ADFTDDataset(BaseDataset):
    task = 'eyesclosed'
    dataset_name = 'ADFTD-RS'
    derivative_directory = 'derivatives'

    def __init__(self, root_path='./ADFTD-RS', chunk_size=400, overlap=0,
                 num_channel=19, sfreq=200, notch_freqs=50., l_freq=.5, h_freq=45.,
                 ref_channels=None, online_transform=None, offline_transform=None,
                 label_transform=None, before_trial=None, after_trial=None,
                 after_subject=None, io_path=None, io_size=10485760, io_mode='lmdb',
                 num_worker=0, verbose=True, preprocessed=False):
        _settings(chunk_size, overlap, num_channel, sfreq,
                  notch_freqs, l_freq, h_freq, ref_channels)
        if not isinstance(preprocessed, bool):
            raise ValueError('preprocessed must be a boolean.')
        params = dict(root_path=root_path, chunk_size=chunk_size, overlap=overlap,
                      num_channel=num_channel, sfreq=sfreq, notch_freqs=notch_freqs,
                      l_freq=l_freq, h_freq=h_freq, ref_channels=ref_channels,
                      online_transform=online_transform, offline_transform=offline_transform,
                      label_transform=label_transform, before_trial=before_trial,
                      after_trial=after_trial, after_subject=after_subject,
                      io_path=io_path if io_path is not None else get_random_dir_path(dir_prefix='datasets'),
                      io_size=io_size, io_mode=io_mode, num_worker=num_worker, verbose=verbose, preprocessed=preprocessed)
        super().__init__(**params)
        self.__dict__.update(params)

    @classmethod
    def set_records(cls, root_path='./ADFTD-RS', preprocessed=False, **kwargs):
        root = Path(root_path)
        if not isinstance(preprocessed, bool):
            raise ValueError('preprocessed must be a boolean.')
        people = _participants(root / 'participants.tsv')
        data_root = root / cls.derivative_directory if preprocessed else root
        records = []
        for subject, group in sorted(people.items()):
            paths = sorted(path for path in (data_root / subject).rglob('*_eeg.set')
                           if path.parent.name == 'eeg' and f'task-{cls.task}' in path.stem.split('_'))
            if not paths:
                raise FileNotFoundError(f'No {cls.task} SET recording for known participant {subject}.')
            for path in paths:
                if not path.name.startswith(subject + '_'):
                    raise ValueError('SET filename and BIDS participant directory must agree.')
                raw_path = root / path.relative_to(data_root)
                channels = raw_path.with_name(raw_path.name.replace('_eeg.set', '_channels.tsv'))
                sidecar = raw_path.with_suffix('.json')
                if not channels.is_file() or not sidecar.is_file():
                    raise FileNotFoundError('Every ADFTD recording needs matching channels.tsv and EEG JSON.')
                session = next((part for part in path.stem.split('_') if part.startswith('ses-')), 'n/a')
                records.append(dict(file_path=str(path), channels_path=str(channels), sidecar_path=str(sidecar),
                                    participant_id=subject, subject_id=int(subject[4:]), group=group,
                                    label=LABELS[group], record_id=f"{cls.dataset_name}_{'preprocessed_' if preprocessed else ''}{path.stem[:-4]}",
                                    session_id=session, raw_file_path=str(raw_path), data_source='preprocessed' if preprocessed else 'raw'))
        return records

    @classmethod
    def read_record(cls, record: Dict, num_channel=19, sfreq=200,
                    notch_freqs=50., l_freq=.5, h_freq=45., ref_channels=None, preprocessed=False, **kwargs):
        _settings(400, 0, num_channel, sfreq, notch_freqs, l_freq, h_freq, ref_channels)
        if not isinstance(preprocessed, bool):
            raise ValueError('preprocessed must be a boolean.')
        path = Path(record['file_path'])
        if (record['group'] not in LABELS or record['label'] != LABELS[record['group']]
                or record['subject_id'] != int(record['participant_id'][4:])
                or not path.name.startswith(record['participant_id'] + '_')
                or record['record_id'] != f"{cls.dataset_name}_{'preprocessed_' if preprocessed else ''}{path.stem[:-4]}"
                or record['data_source'] != ('preprocessed' if preprocessed else 'raw')
                or f'task-{cls.task}' not in path.stem.split('_')):
            raise ValueError('ADFTD record identity and diagnosis metadata must agree.')
        raw_path = Path(record['raw_file_path'])
        if (raw_path.name != path.name
                or Path(record['channels_path']) != raw_path.with_name(raw_path.name.replace('_eeg.set', '_channels.tsv'))
                or Path(record['sidecar_path']) != raw_path.with_suffix('.json')):
            raise ValueError('ADFTD source sidecars must match the SET recording.')
        with Path(record['channels_path']).open(encoding='utf-8-sig', newline='') as file:
            reader = csv.DictReader(file, delimiter='\t')
            if not {'name', 'type', 'units'} <= set(reader.fieldnames or []):
                raise ValueError('channels.tsv needs name, type and units columns.')
            channel_rows = list(reader)
        if (len(channel_rows) != 19 or {row['name'].upper() for row in channel_rows} != set(CHANNELS)
                or any(row['type'] not in ('EEG', 'n/a') or row['units'] not in ('microV', 'uV', 'µV', 'n/a') for row in channel_rows)):
            raise ValueError('ADFTD channel metadata must describe the 19 EEG columns in microvolts or n/a.')
        metadata = json.loads(Path(record['sidecar_path']).read_text(encoding='utf-8-sig'))
        if (metadata.get('SamplingFrequency') != 500 or metadata.get('TaskName') != cls.task
                or metadata.get('EEGChannelCount', 19) != 19 or metadata.get('RecordingType', 'continuous') != 'continuous'):
            raise ValueError('ADFTD EEG JSON must declare the matching task and source sampling rate.')
        values, cuts = _load_set(path)
        duration = metadata.get('RecordingDuration')
        if (isinstance(duration, bool) or not isinstance(duration, Real) or not np.isfinite(duration)
                or duration <= 0 or values.shape[1] / 500 - duration > 1 / 500 + 1e-9
                or (not preprocessed and abs(duration - values.shape[1] / 500) > 1 / 500 + 1e-9)):
            raise ValueError('EEG JSON duration must agree with the SET sample count.')
        segments = []
        for index, (first, last) in enumerate(zip(cuts, cuts[1:])):
            eeg = values[:, first:last].copy()
            if sfreq != 500 or any(value is not None for value in (notch_freqs, l_freq, h_freq, ref_channels)):
                raw = mne.io.RawArray(eeg, mne.create_info(CHANNELS, 500, 'eeg'), verbose=False)
                if notch_freqs is not None:
                    raw.notch_filter(notch_freqs, verbose=False)
                if l_freq is not None or h_freq is not None:
                    raw.filter(l_freq, h_freq, verbose=False)
                if ref_channels is not None:
                    raw.set_eeg_reference(ref_channels, verbose=False)
                if sfreq != 500:
                    raw.resample(sfreq, verbose=False)
                eeg = raw.get_data()
            segments.append(dict(eeg=eeg[:num_channel].copy(), source_first=first, source_last=last, index=index))
        return dict(segments=segments, sfreq=sfreq,
                    channel_units_metadata=json.dumps({row['name'].upper(): row['units'] for row in channel_rows}))

    @classmethod
    def fake_record(cls, num_channel=19, sfreq=200, **kwargs):
        _settings(400, 0, num_channel, sfreq)
        record = dict(participant_id='sub-001', subject_id=1, group='A', label=1,
                      record_id=f'{cls.dataset_name}_sub-001_task-{cls.task}', session_id='n/a', data_source='raw')
        return dict(record=record, segments=[dict(eeg=np.random.default_rng(42).standard_normal((num_channel, 8 * sfreq)),
                    source_first=0, source_last=4000, index=0)], sfreq=sfreq, channel_units_metadata=json.dumps({channel: 'n/a' for channel in CHANNELS}))

    @classmethod
    def process_record(cls, record: Dict, segments, sfreq=200, chunk_size=400,
                       overlap=0, num_channel=19, before_trial=None, offline_transform=None,
                       channel_units_metadata='{}', **kwargs):
        _settings(chunk_size, overlap, num_channel, sfreq)
        if (record['group'] not in LABELS or record['label'] != LABELS[record['group']] or not segments
                or not re.fullmatch(r'sub-[0-9]+', record['participant_id'])
                or record['subject_id'] != int(record['participant_id'][4:])
                or record.get('data_source') not in ('raw', 'preprocessed')):
            raise ValueError('Require ADFTD signal segments and a matching diagnosis label.')
        output_offset = 0
        for segment in segments:
            values = np.asarray(segment['eeg'])
            if (values.ndim != 2 or values.shape[0] < num_channel or values.shape[1] == 0
                    or values.dtype.kind not in 'iuf' or not np.isfinite(values).all()):
                raise ValueError('Require finite nonempty ADFTD channel/time segments.')
            expected = round((segment['source_last'] - segment['source_first']) * sfreq / 500)
            if values.shape[1] != expected or segment['source_first'] < 0 or segment['source_last'] <= segment['source_first']:
                raise ValueError('ADFTD segment sample counts must match the source/output clocks.')
            trial = values[:num_channel].copy()
            if before_trial is not None:
                trial = np.asarray(before_trial(trial))
                if trial.shape != (num_channel, values.shape[1]) or trial.dtype.kind not in 'iuf' or not np.isfinite(trial).all():
                    raise ValueError('before_trial must preserve finite segment geometry.')
            size = trial.shape[1] if chunk_size <= 0 else chunk_size
            if overlap >= size:
                raise ValueError('Overlap must be smaller than each selected chunk.')
            identity = f"{record['record_id']}_segment-{segment['index']}"
            for start in range(0, trial.shape[1] - size + 1, size - overlap):
                end = start + size
                clip = trial[:, start:end].copy()
                if offline_transform is not None:
                    clip = offline_transform(eeg=clip)['eeg']
                key = f'{identity}_{start}'
                info = dict(subject_id=record['subject_id'], participant_id=record['participant_id'],
                            session_id=record['session_id'], record_id=record['record_id'], trial_id=identity,
                            segment_index=segment['index'], group=record['group'], group_name=GROUP_NAMES[record['group']],
                            label=record['label'], task=cls.task, data_source=record['data_source'],
                            source_clock='stored_recording', clip_id=key, start_at=output_offset + start,
                            end_at=output_offset + end, sfreq=sfreq, source_sfreq=500, signal_unit='V',
                            units_basis='EEGLAB_microvolt_format_convention', channel_units_metadata=channel_units_metadata,
                            source_start_seconds=segment['source_first'] / 500 + start / sfreq,
                            source_end_seconds=min(segment['source_last'] / 500, segment['source_first'] / 500 + end / sfreq),
                            channel_names=json.dumps(CHANNELS[:num_channel]))
                yield dict(eeg=clip, key=key, info=info)
            output_offset += trial.shape[1]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        info = self.read_info(index)
        eeg = self.read_eeg(str(info['_record_id']), str(info['clip_id']))
        signal = self.online_transform(eeg=eeg)['eeg'] if self.online_transform else eeg
        label = self.label_transform(y=info)['y'] if self.label_transform else info
        return signal, label

    @property
    def repr_body(self) -> Dict:
        names = ('root_path', 'chunk_size', 'overlap', 'num_channel', 'sfreq',
                 'notch_freqs', 'l_freq', 'h_freq', 'ref_channels', 'online_transform',
                 'offline_transform', 'label_transform', 'before_trial', 'after_trial', 'after_subject', 'preprocessed')
        return dict(super().repr_body, **{name: getattr(self, name) for name in names})
