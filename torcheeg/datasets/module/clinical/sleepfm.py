"""Patient-separated, explicitly described SleepFM survival embedding cohorts."""

import copy
import csv
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


_SCHEMA_FIELDS = {
    'endpoint_names',
    'demographic_names',
    'modality_names',
    'embedding_step_seconds',
    'embedding_width',
    'upstream_source_revision',
    'upstream_checkpoint_sha256',
    'preprocessing_id',
    'demographic_encoding_id',
    'event_time_unit',
    'data_kind',
}
_RECORD_FIELDS = {'study_id', 'subject_id', 'path', 'sha256', 'split'}
_SAMPLE_FIELDS = {
    'embeddings',
    'padding_mask',
    'demographics',
    'event_times',
    'is_event',
    'study_id',
    'subject_id',
    'schema_fingerprint',
}


def _text(value, name):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            name + ' must be a nonempty string without surrounding whitespace.'
        )
    return value


def _digest(value, length, name):
    if (
        not isinstance(value, str)
        or re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is None
    ):
        raise ValueError(name + ' must be a lowercase hexadecimal identity.')
    return value


def _names(values, count, name):
    if (
        not isinstance(values, list)
        or (count is not None and len(values) != count)
        or not values
    ):
        raise ValueError(name + ' must declare every ordered column.')
    for value in values:
        _text(value, name)
    if len(set(values)) != len(values) or 'Study ID' in values:
        raise ValueError(name + ' must contain distinct names separate from Study ID.')
    return tuple(values)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field: ' + key)
        result[key] = value
    return result


def _schema(schema):
    if not isinstance(schema, dict) or set(schema) != _SCHEMA_FIELDS:
        raise ValueError('Require the complete explicit survival embedding schema.')
    endpoints = _names(schema['endpoint_names'], 1065, 'endpoint_names')
    demographics = _names(schema['demographic_names'], 2, 'demographic_names')
    modalities = _names(schema['modality_names'], None, 'modality_names')
    if any(name not in ('BAS', 'RESP', 'EKG', 'EMG') for name in modalities):
        raise ValueError('Require native BAS/RESP/EKG/EMG modality names.')
    if (
        type(schema['embedding_step_seconds']) is not int
        or schema['embedding_step_seconds'] != 5
        or type(schema['embedding_width']) is not int
        or schema['embedding_width'] != 128
    ):
        raise ValueError('Require the native 5-second/128-wide embedding convention.')
    _digest(schema['upstream_source_revision'], 40, 'upstream_source_revision')
    _digest(schema['upstream_checkpoint_sha256'], 64, 'upstream_checkpoint_sha256')
    for name in ('preprocessing_id', 'demographic_encoding_id', 'event_time_unit'):
        _text(schema[name], name)
    if schema['data_kind'] not in ('measured', 'synthetic'):
        raise ValueError('Declare data_kind as measured or synthetic.')
    return endpoints, demographics, modalities


def _table(path, names, selected, kind):
    payload = Path(path).read_bytes()
    rows = csv.reader(io.StringIO(payload.decode('utf-8-sig')))
    header = next(rows, None)
    expected = {'Study ID', *names}
    if header is None or len(header) != len(expected) or set(header) != expected:
        raise ValueError(
            kind + ' must contain Study ID and exactly the declared named columns.'
        )
    id_position = header.index('Study ID')
    positions = [header.index(name) for name in names]
    seen, result = set(), {}
    for row in rows:
        if len(row) != len(header):
            raise ValueError(kind + ' has an incomplete or extra CSV field.')
        study = _text(row[id_position], 'Study ID')
        if study in seen:
            raise ValueError(kind + ' contains duplicate Study ID rows.')
        seen.add(study)
        if study not in selected:
            continue
        try:
            values = np.asarray([float(row[i]) for i in positions], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                kind + ' requires explicit numeric values, not missing outcomes.'
            ) from error
        if not np.isfinite(values).all():
            raise ValueError(kind + ' requires finite selected values.')
        if kind == 'event_times':
            if (values < 0).any():
                raise ValueError('Follow-up times must be nonnegative.')
        elif kind == 'is_event':
            if not np.isin(values, [0, 1]).all():
                raise ValueError(
                    'Events must be explicit zero/one censoring indicators.'
                )
            values = values.astype(np.bool_)
        else:
            if (np.abs(values) > np.finfo(np.float32).max).any():
                raise ValueError('Encoded demographics must be finite in float32.')
            values = values.astype(np.float32)
        result[study] = torch.from_numpy(values.copy())
    if set(result) != selected:
        raise ValueError(kind + ' lacks selected Study ID rows.')
    return result, hashlib.sha256(payload).hexdigest()


class SleepFMSurvivalDataset(Dataset):
    r"""Read an explicit patient-separated cohort of native SleepFM embeddings.

    Each baseline study has one HDF5 file with modality datasets [steps,128].
    Manifest schema declares the complete ordered 1065 endpoints, two already
    encoded demographic columns, modality order, 5-second embedding convention,
    source/checkpoint/preprocessing identities, time unit and data kind.
    It declares one selected baseline per subject and train/validation/test
    membership. Study IDs and subject IDs remain strings, including leading zeros.

    Outcome/demographic CSVs contain ``Study ID`` and exactly their declared
    columns. They are aligned by name and ID, never by row/column position or
    numeric conversion of identifiers. Missing follow-up is not censoring;
    missing endpoint columns are not filled with zeros. No demographic codec
    or data-dependent normalization is fitted by this reader.

    Each record's SHA256 binds the exact HDF5 bytes read. External/soft links,
    virtual datasets and external storage are rejected. The first declared
    modality must exist; absent optional modalities are explicitly masked.
    Present modalities must share a positive length no greater than 6480.
    Nothing is cropped, resampled or inferred from file names. Files are read
    lazily; :meth:`validate_files` explicitly checks all selected recordings.

    Args:
        manifest_path (str or Path): Version-one schema/record JSON manifest.
        event_times_path (str or Path): Complete named follow-up-time CSV.
        is_event_path (str or Path): Complete named binary-event CSV.
        demographics_path (str or Path): Two explicitly encoded columns CSV.
        split (str): Declared train, validation or test split.
            (default: :obj:`'train'`)

    A sample is a dictionary of embeddings, padding_mask, demographics,
    event_times, is_event, study_id, subject_id and schema_fingerprint. Use
    :func:`collate_sleepfm_survival` for variable-length batches. The metadata
    table ``info`` has one row per selected baseline. These are caller-supplied
    cohorts, not a bundled clinical dataset or verified diagnosis benchmark.
    """

    def __init__(
        self,
        manifest_path,
        event_times_path,
        is_event_path,
        demographics_path,
        split='train',
    ):
        if split not in ('train', 'validation', 'test'):
            raise ValueError('Require a declared train, validation or test split.')
        path = Path(manifest_path).resolve()
        payload = path.read_bytes()
        manifest = json.loads(payload, object_pairs_hook=_unique_object)
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {'version', 'schema', 'records'}
            or type(manifest['version']) is not int
            or manifest['version'] != 1
        ):
            raise ValueError('Require a complete version-one cohort manifest.')
        self._endpoint_names, self._demographic_names, self._modality_names = _schema(
            manifest['schema']
        )
        records = manifest['records']
        if not isinstance(records, list) or not records:
            raise ValueError(
                'Declare the selected baseline studies and subject-separated splits.'
            )
        studies, subjects, paths, hashes = set(), set(), set(), set()
        selected = []
        for record in records:
            if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
                raise ValueError(
                    'Require study_id, subject_id, path, sha256 and split for every record.'
                )
            study = _text(record['study_id'], 'study_id')
            subject = _text(record['subject_id'], 'subject_id')
            filename = _text(record['path'], 'path')
            checksum = _digest(record['sha256'], 64, 'sha256')
            source = (path.parent / filename).resolve()
            if record['split'] not in ('train', 'validation', 'test'):
                raise ValueError('Every record must use a declared cohort split.')
            if study in studies or subject in subjects:
                raise ValueError(
                    'Select exactly one baseline per study/subject, without split overlap.'
                )
            if source in paths or checksum in hashes:
                raise ValueError(
                    'Do not reuse an embedding file or identical file contents across patients.'
                )
            studies.add(study)
            subjects.add(subject)
            paths.add(source)
            hashes.add(checksum)
            if record['split'] == split:
                selected.append(
                    dict(
                        study_id=study,
                        subject_id=subject,
                        split=split,
                        path=str(source),
                        sha256=checksum,
                    )
                )
        if not selected:
            raise ValueError('The requested cohort split has no selected baselines.')
        self.split = split
        self.info = pd.DataFrame(selected)
        self._records = {record['study_id']: record for record in selected}
        ids = set(self._records)
        self._times, times_sha = _table(
            event_times_path, self.endpoint_names, ids, 'event_times'
        )
        self._events, events_sha = _table(
            is_event_path, self.endpoint_names, ids, 'is_event'
        )
        self._demographics, demographics_sha = _table(
            demographics_path, self.demographic_names, ids, 'demographics'
        )
        self._schema = copy.deepcopy(manifest['schema'])
        self._schema_fingerprint = hashlib.sha256(
            json.dumps(self._schema, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()
        self._provenance = dict(
            version=1,
            schema=self._schema,
            manifest_sha256=hashlib.sha256(payload).hexdigest(),
            event_times_sha256=times_sha,
            is_event_sha256=events_sha,
            demographics_sha256=demographics_sha,
        )

    @property
    def endpoint_names(self):
        return self._endpoint_names

    @property
    def demographic_names(self):
        return self._demographic_names

    @property
    def modality_names(self):
        return self._modality_names

    @property
    def schema(self):
        return copy.deepcopy(self._schema)

    @property
    def provenance(self):
        return copy.deepcopy(self._provenance)

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        import h5py

        row = self.info.iloc[index]
        record = self._records[row['study_id']]
        if any(row[name] != value for name, value in record.items()):
            raise ValueError(
                'The selected metadata row no longer matches its declared patient.'
            )
        payload = Path(record['path']).read_bytes()
        if hashlib.sha256(payload).hexdigest() != record['sha256']:
            raise ValueError('Embedding file contents differ from the declared SHA256.')
        arrays = {}
        # Hash and decode the same immutable snapshot, without external files.
        with h5py.File(io.BytesIO(payload), 'r') as source:
            if (
                not set(source).issubset(self.modality_names)
                or self.modality_names[0] not in source
            ):
                raise ValueError(
                    'Require the first modality and no undeclared embedding datasets.'
                )
            for name in source:
                if not isinstance(source.get(name, getlink=True), h5py.HardLink):
                    raise ValueError(
                        'Embedding data must not use soft or external links.'
                    )
                value = source[name]
                if (
                    not isinstance(value, h5py.Dataset)
                    or value.is_virtual
                    or value.external
                ):
                    raise ValueError(
                        'Require ordinary self-contained HDF5 modality datasets.'
                    )
                if (
                    value.ndim != 2
                    or value.shape[1] != 128
                    or not 1 <= value.shape[0] <= 6480
                    or value.dtype.kind != 'f'
                    or value.dtype.itemsize != 4
                ):
                    raise ValueError(
                        'Require complete float32 modality arrays[1..6480,128].'
                    )
                arrays[name] = np.asarray(value, dtype=np.float32)
                if not np.isfinite(arrays[name]).all():
                    raise ValueError('Require finite pretrained embedding values.')
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1:
            raise ValueError(
                'Present modalities must share aligned five-second boundaries and lengths.'
            )
        steps = lengths.pop()
        x = torch.zeros(
            len(self.modality_names), steps, 128, dtype=torch.float32, device='cpu'
        )
        mask = torch.ones(
            len(self.modality_names), steps, dtype=torch.bool, device='cpu'
        )
        for channel, name in enumerate(self.modality_names):
            if name in arrays:
                x[channel] = torch.from_numpy(arrays[name])
                mask[channel] = False
        study = record['study_id']
        return dict(
            embeddings=x,
            padding_mask=mask,
            demographics=self._demographics[study].clone(),
            event_times=self._times[study].clone(),
            is_event=self._events[study].clone(),
            study_id=study,
            subject_id=record['subject_id'],
            schema_fingerprint=self._schema_fingerprint,
        )

    def validate_files(self):
        r"""Validate every selected file/hash/array before starting an experiment."""
        for index in range(len(self)):
            self[index]


def collate_sleepfm_survival(batch):
    r"""Pad native survival embeddings, preserving named schema and patient order.

    Return tensors [B,C,max_steps,128], mask[B,C,max_steps], two demographics,
    complete times/events[B,1065], study/subject ID lists and one schema identity.
    Reject mixed schemas and repeated patients within a risk-set batch. No
    tensor is cropped; padded modalities and time steps are masked explicitly.
    """
    if not isinstance(batch, (tuple, list)) or not batch:
        raise ValueError('Require a nonempty sequence of survival cohort samples.')
    for item in batch:
        if not isinstance(item, dict) or set(item) != _SAMPLE_FIELDS:
            raise ValueError('Require complete survival cohort sample dictionaries.')
    first = batch[0]['embeddings']
    if not isinstance(first, torch.Tensor) or first.ndim != 3:
        raise ValueError('Require modality embeddings[C,steps,128].')
    channels = first.shape[0]
    schemas, studies, subjects = set(), set(), set()
    for item in batch:
        x, mask = item['embeddings'], item['padding_mask']
        if (
            not isinstance(x, torch.Tensor)
            or x.device.type != 'cpu'
            or x.dtype != torch.float32
            or x.ndim != 3
            or x.shape[0] != channels
            or not 1 <= channels <= 4
            or x.shape[2] != 128
            or not 1 <= x.shape[1] <= 6480
            or not torch.isfinite(x).all()
        ):
            raise ValueError('Require finite CPU float32 embeddings[C,1..6480,128].')
        if (
            not isinstance(mask, torch.Tensor)
            or mask.device.type != 'cpu'
            or mask.dtype != torch.bool
            or mask.shape != x.shape[:2]
            or mask[0, 0]
            or not torch.equal(mask, mask[:, :1].expand_as(mask))
        ):
            raise ValueError(
                'Sample masks must represent whole missing modalities with modality0 present.'
            )
        for name, size, dtype in [
            ('demographics', 2, torch.float32),
            ('event_times', 1065, torch.float64),
            ('is_event', 1065, torch.bool),
        ]:
            value = item[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != 'cpu'
                or value.shape != (size,)
                or value.dtype != dtype
                or not torch.isfinite(value).all()
                or (name == 'event_times' and (value < 0).any())
            ):
                raise ValueError(
                    'Require complete finite typed survival targets and demographics.'
                )
        schema = _digest(item['schema_fingerprint'], 64, 'schema_fingerprint')
        study, subject = (
            _text(item['study_id'], 'study_id'),
            _text(item['subject_id'], 'subject_id'),
        )
        if study in studies or subject in subjects:
            raise ValueError(
                'Do not repeat a study or patient within one survival risk set.'
            )
        schemas.add(schema)
        studies.add(study)
        subjects.add(subject)
    if len(schemas) != 1:
        raise ValueError(
            'Do not combine different endpoint/encoding/modality schemas in one batch.'
        )
    steps = max(item['embeddings'].shape[1] for item in batch)
    x = torch.zeros(len(batch), channels, steps, 128, dtype=torch.float32, device='cpu')
    mask = torch.ones(len(batch), channels, steps, dtype=torch.bool, device='cpu')
    for row, item in enumerate(batch):
        length = item['embeddings'].shape[1]
        x[row, :, :length] = item['embeddings']
        mask[row, :, :length] = item['padding_mask']
    return dict(
        embeddings=x,
        padding_mask=mask,
        demographics=torch.stack([item['demographics'] for item in batch]),
        event_times=torch.stack([item['event_times'] for item in batch]),
        is_event=torch.stack([item['is_event'] for item in batch]),
        study_id=[item['study_id'] for item in batch],
        subject_id=[item['subject_id'] for item in batch],
        schema_fingerprint=next(iter(schemas)),
    )
