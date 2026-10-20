"""
Run the complete external SleepFM diagnosis head
================================================

Use separately obtained CC-BY-NC4.0 SleepFM-Clinical source/checkpoints at
2bcbae04c3592f61352addb7ac3d4193f0a3ca25. This explicit local-backend example
loads every original diagnosis tensor, including both demographic columns
and all 1065 Cox log-risk outputs. It does not convert them to probabilities,
assign a new diagnosis, fit a survival baseline or report clinical accuracy.

Supply a local tensor-only payload with features[B,modalities,steps,128],
boolean padding_mask[B,modalities,steps], demographics[B,2] already encoded
in the original training convention, and metadata_json. That JSON must
contain ordered modality_names, demographic_names, record_ids, explicit
preprocessing and demographic_encoding descriptions, an explicit synthetic
boolean and feature_encoder provenance. Contextual five-second features
must come from the complete pinned upstream checkpoint at 128 Hz,640samples
per patch. Obtain each modality's tokens by calling SleepFM(...,
return_tokens=True) and stack those tokens in the declared modality order.
Do not pass raw EEG as embeddings or raw ages as already encoded columns.
The supplied metadata are checked and saved; a description does not verify
the underlying training codec. Use a validated clinical preprocessing and
encoding protocol before interpreting results. The author demonstration
PSG/demographics/disease files are explicitly synthetic, not clinical data.

The official label_mapping.csv is obtained separately with the backend.
Its complete label_idx order is validated and retained with output metadata;
no disease names are guessed. Save the entire head state, provenance,
label mapping and log risks, then strictly restore and reproduce outputs.
This example performs inference. Real clinical survival training requires
appropriate event/censoring labels and a separate evaluation protocol.

Run::

    python examples/examples_sleepfm_diagnosis.py --input-path ./embeddings.pt \\
        --backend-root ./sleepfm-clinical --output-path ./sleepfm-log-risk.pt"""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import torch

from torcheeg.models import SleepFMDiagnosis


SOURCE_REVISION = '2bcbae04c3592f61352addb7ac3d4193f0a3ca25'
UPSTREAM_SHA256 = 'ffc9fc10233ebc4d0aae71abce87070db51b8bac6e4b16a5c1b4401a5f73f799'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_diagnosis(root, modality_names, demographic_names):
    root = Path(root).resolve()
    revision = subprocess.run(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError('Use the documented pinned complete SleepFM author checkout.')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)
    module_name = (
        'sleepfm_author_' + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    )
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            module_name, root / 'sleepfm/models/models.py'
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
    config = json.loads(
        (root / 'sleepfm/checkpoints/model_diagnosis/config.json').read_text()
    )
    return SleepFMDiagnosis.from_pretrained(
        root / 'sleepfm/checkpoints/model_diagnosis/best.pth',
        module.DiagnosisFinetuneFullLSTMCOXPHWithDemo(**config['model_params']),
        modality_names,
        demographic_names,
    )


def load_input(path):
    payload = torch.load(path, weights_only=True, map_location='cpu')
    required = {'features', 'padding_mask', 'demographics', 'metadata_json'}
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or not isinstance(payload['metadata_json'], str)
    ):
        raise ValueError('Require the documented complete tensor-only input payload.')
    metadata = json.loads(payload['metadata_json'])
    if not isinstance(metadata, dict):
        raise ValueError('Input metadata must be a JSON object.')
    for key in ['preprocessing', 'demographic_encoding']:
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(
                'Describe the actual preprocessing and demographic encoding explicitly.'
            )
    if not isinstance(metadata.get('synthetic'), bool):
        raise ValueError('Declare whether the input data are synthetic.')
    encoder = metadata.get('feature_encoder', {})
    expected = dict(
        revision=SOURCE_REVISION,
        checkpoint_sha256=UPSTREAM_SHA256,
        sampling_rate=128,
        patch_samples=640,
        representation='contextual_5s_tokens',
    )
    if encoder != expected:
        raise ValueError(
            'Features must declare the complete pinned native upstream token encoder.'
        )
    features = payload['features']
    records = metadata.get('record_ids')
    if (
        not isinstance(features, torch.Tensor)
        or features.ndim != 4
        or not features.is_floating_point()
        or not torch.isfinite(features).all()
    ):
        raise ValueError(
            'Require finite native contextual features[B,modalities,steps,128].'
        )
    if (
        not isinstance(records, list)
        or len(records) != features.shape[0]
        or not all(isinstance(v, str) and v for v in records)
        or len(set(records)) != len(records)
    ):
        raise ValueError('Provide a distinct explicit record ID for every example.')
    for key in ['modality_names', 'demographic_names']:
        if (
            not isinstance(metadata.get(key), list)
            or not metadata[key]
            or not all(isinstance(v, str) and v for v in metadata[key])
        ):
            raise ValueError('Provide explicit ordered modality and demographic names.')
    return payload, metadata


def read_labels(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1065 or any(
        not {'label_idx', 'phecode', 'phenotype'} <= set(row) for row in rows
    ):
        raise ValueError(
            'Require the complete separately supplied1065-row official label mapping.'
        )
    rows = sorted(rows, key=lambda row: int(row['label_idx']))
    if [int(row['label_idx']) for row in rows] != list(range(1065)):
        raise ValueError(
            'Disease output indices must be complete, distinct and consecutive.'
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Run the complete separately supplied native SleepFM Cox log-risk head.'
    )
    parser.add_argument('--input-path', required=True)
    parser.add_argument('--backend-root', required=True)
    parser.add_argument('--output-path', default='./sleepfm-log-risk.pt')
    args = parser.parse_args()
    payload, metadata = load_input(args.input_path)
    labels_path = Path(args.backend_root) / 'sleepfm/configs/label_mapping.csv'
    labels = read_labels(labels_path)
    model = build_diagnosis(
        args.backend_root, metadata['modality_names'], metadata['demographic_names']
    ).eval()
    with torch.no_grad():
        output = model(
            payload['features'], payload['demographics'], payload['padding_mask']
        )
    if not torch.isfinite(output).all():
        raise ValueError('Log-risk output must be finite.')
    provenance = dict(
        input_metadata=metadata,
        input_sha256=sha256(args.input_path),
        checkpoint_sha256=sha256(
            Path(args.backend_root) / 'sleepfm/checkpoints/model_diagnosis/best.pth'
        ),
        label_mapping_sha256=sha256(labels_path),
        label_mapping=labels,
        backend_revision=SOURCE_REVISION,
        weights_license='CC-BY-NC-4.0',
        output='original Cox log-risk columns; not calibrated probabilities',
        clinical_validation=False,
    )
    target = Path(args.output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            state_dict=model.state_dict(),
            metadata_json=json.dumps(provenance),
            log_risk=output,
        ),
        target,
    )
    saved = torch.load(target, weights_only=True, map_location='cpu')
    restored = build_diagnosis(
        args.backend_root, metadata['modality_names'], metadata['demographic_names']
    ).eval()
    restored.load_state_dict(saved['state_dict'], strict=True)
    with torch.no_grad():
        torch.testing.assert_close(
            restored(
                payload['features'], payload['demographics'], payload['padding_mask']
            ),
            output,
            rtol=0,
            atol=0,
        )
    if sha256(args.input_path) != provenance['input_sha256']:
        raise ValueError('Input artifact changed during inference.')
    print(
        json.dumps(
            dict(
                artifact=str(target),
                records=len(output),
                output_columns=1065,
                synthetic=metadata['synthetic'],
                clinical_validation=False,
                strict_restore=True,
            )
        )
    )


if __name__ == '__main__':
    main()
