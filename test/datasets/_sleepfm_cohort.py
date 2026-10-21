"""Synthetic file/schema fixtures only; no real clinical signals or outcomes."""

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def create_cohort(
    directory,
    endpoint_names=None,
    upstream_sha='a' * 64,
    source_revision='b' * 40,
    counts=(6, 3, 2),
):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    names = (
        endpoint_names
        if endpoint_names is not None
        else ['synthetic_' + str(i) for i in range(1065)]
    )
    schema = dict(
        endpoint_names=names,
        demographic_names=['encoded_age', 'encoded_gender'],
        modality_names=['BAS', 'RESP', 'EKG', 'EMG'],
        embedding_step_seconds=5,
        embedding_width=128,
        upstream_source_revision=source_revision,
        upstream_checkpoint_sha256=upstream_sha,
        preprocessing_id='synthetic_fixture_no_PSG',
        demographic_encoding_id='synthetic_fixture_no_patient_demographics',
        event_time_unit='synthetic_day',
        data_kind='synthetic',
    )
    records, expected = [], {}
    for split, count in zip(['train', 'validation', 'test'], counts):
        for _ in range(count):
            index = len(records)
            study = f'{index:04d}'
            filename = root / (study + '.hdf5')
            steps = 2 + index % 3
            arrays = {}
            with h5py.File(filename, 'w') as source:
                for channel, modality in enumerate(schema['modality_names']):
                    if (modality == 'RESP' and index % 2 == 0) or (
                        modality == 'EMG' and index % 3 == 0
                    ):
                        continue
                    values = np.sin(
                        np.arange(steps * 128, dtype=np.float32) / 31 + index + channel
                    ).reshape(steps, 128)
                    arrays[modality] = values
                    source.create_dataset(modality, data=values)
            times = np.arange(1065, dtype=np.float64) / 10000 + index + 1
            events = (np.arange(1065) + index) % 3 != 0
            demographics = np.array([index / 100, index % 2], dtype=np.float32)
            records.append(
                dict(
                    study_id=study,
                    subject_id='patient_' + study,
                    path=filename.name,
                    sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
                    split=split,
                )
            )
            expected[study] = dict(
                arrays=arrays, times=times, events=events, demographics=demographics
            )
    manifest = dict(version=1, schema=schema, records=records)
    manifest_path = root / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    paths = dict(manifest_path=manifest_path)
    for kind, columns in [
        ('event_times', names),
        ('is_event', names),
        ('demographics', schema['demographic_names']),
    ]:
        path = root / (kind + '.csv')
        # Reverse both column and row order to exercise named joins explicitly.
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(list(reversed(columns)) + ['Study ID'])
            for record in reversed(records):
                key = {
                    'event_times': 'times',
                    'is_event': 'events',
                    'demographics': 'demographics',
                }[kind]
                values = expected[record['study_id']][key]
                if kind == 'is_event':
                    values = values.astype(int)
                writer.writerow(list(reversed(values.tolist())) + [record['study_id']])
        paths[
            {
                'event_times': 'event_times_path',
                'is_event': 'is_event_path',
                'demographics': 'demographics_path',
            }[kind]
        ] = path
    return paths, manifest, expected


def save_manifest(paths, manifest):
    Path(paths['manifest_path']).write_text(json.dumps(manifest, indent=2) + '\n')


def replace_embedding(paths, manifest, position, writer):
    record = manifest['records'][position]
    path = Path(paths['manifest_path']).parent / record['path']
    with h5py.File(path, 'w') as source:
        writer(source)
    record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_manifest(paths, manifest)
