"""Complete official-model CLI and independent raw-file training oracle.

The generated multi-subject cohort and author demo are explicitly synthetic.
They establish file/protocol correctness, not measured clinical performance.
"""

import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import torch

from torcheeg.datasets import SleepFMSurvivalDataset, collate_sleepfm_survival
from torcheeg.models import SleepFMSurvivalObjective
from test.datasets._sleepfm_cohort import create_cohort, save_manifest
from test.models.test_sleepfm_cohort_example import load_workflow, assert_payload_equal


def csv_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return {row['Study ID']: row for row in csv.DictReader(stream)}


def raw_batch(directory, records, schema, tables):
    """Read HDF5 and named CSVs without the reader or collator under test."""
    arrays, lengths = [], []
    for record in records:
        with h5py.File(directory / record['path'], 'r') as source:
            data = {name: source[name][:] for name in source}
        arrays.append(data)
        lengths.append(len(next(iter(data.values()))))
    x = torch.zeros(len(records), len(schema['modality_names']), max(lengths), 128)
    mask = torch.ones(x.shape[:3], dtype=torch.bool)
    for row, (data, length) in enumerate(zip(arrays, lengths)):
        for channel, name in enumerate(schema['modality_names']):
            if name in data:
                x[row, channel, :length] = torch.tensor(data[name])
                mask[row, channel, :length] = False
    times, events, demographics = [], [], []
    for record in records:
        study = record['study_id']
        times.append(
            [
                float(tables['event_times'][study][name])
                for name in schema['endpoint_names']
            ]
        )
        events.append(
            [
                bool(int(tables['is_event'][study][name]))
                for name in schema['endpoint_names']
            ]
        )
        demographics.append(
            [
                float(tables['demographics'][study][name])
                for name in schema['demographic_names']
            ]
        )
    return (
        x,
        mask,
        torch.tensor(demographics, dtype=torch.float32),
        torch.tensor(times, dtype=torch.float64),
        torch.tensor(events),
    )


@unittest.skipUnless(
    os.environ.get('TORCHEEG_SLEEPFM_BACKEND_ROOT'),
    'Explicit pinned full SleepFM source and official checkpoints required.',
)
class TestSleepFMCohortNative(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        self.root = Path(os.environ['TORCHEEG_SLEEPFM_BACKEND_ROOT']).resolve()
        self.example = load_workflow()
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.set_num_threads(2)
        with (self.root / 'sleepfm/configs/label_mapping.csv').open() as stream:
            self.endpoint_names = [row['phecode'] for row in csv.DictReader(stream)]
        self.upstream_sha = self.example.SOURCE_FILES[
            'sleepfm/checkpoints/model_base/best.pt'
        ]

    def create(self, directory, counts=(6, 3, 2)):
        return create_cohort(
            directory,
            endpoint_names=self.endpoint_names,
            upstream_sha=self.upstream_sha,
            source_revision=self.example.SOURCE_REVISION,
            counts=counts,
        )

    def test_actual_cli_all_payload_resume_and_independent_original_two_step_update(
        self,
    ):
        paths, manifest, _ = self.create(self.work / 'data')
        product = Path(__file__).resolve().parents[2]
        base = [
            sys.executable,
            str(product / 'examples/examples_sleepfm_survival_training.py'),
            '--backend-root',
            str(self.root),
        ]
        for name, value in paths.items():
            base += ['--' + name.replace('_', '-'), str(value)]
        for name, args in [
            ('step2', ['--steps', '2']),
            ('resumed', ['--steps', '3', '--resume', str(self.work / 'step2.pt')]),
            ('continuous', ['--steps', '3']),
        ]:
            result = subprocess.run(
                base + args + ['--output-path', str(self.work / (name + '.pt'))],
                cwd=product,
                env=dict(
                    os.environ,
                    PYTHONPATH=str(product),
                    OMP_NUM_THREADS='2',
                    MPLBACKEND='Agg',
                ),
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"data_kind": "synthetic"', result.stdout)
        saved = torch.load(
            self.work / 'step2.pt', weights_only=True, map_location='cpu'
        )
        resumed = torch.load(
            self.work / 'resumed.pt', weights_only=True, map_location='cpu'
        )
        continuous = torch.load(
            self.work / 'continuous.pt', weights_only=True, map_location='cpu'
        )
        assert_payload_equal(resumed, continuous)
        self.assertEqual(len(resumed['state_dict']), 37)
        self.assertEqual(resumed['cursor']['epoch'], 1)
        self.assertEqual(resumed['evaluation']['study_id'], ['0006', '0007', '0008'])

        # Build the untouched original35-state model, not the integration
        # factory. Recreate both original AdamW updates from raw named tables.
        source = self.root / 'sleepfm/models/models.py'
        spec = importlib.util.spec_from_file_location(
            'sleepfm_cohort_raw_native', source
        )
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        config = json.loads(
            (self.root / 'sleepfm/checkpoints/model_diagnosis/config.json').read_text()
        )
        torch.manual_seed(42)
        native = original.DiagnosisFinetuneFullLSTMCOXPHWithDemo(
            **config['model_params']
        ).train()
        state = torch.load(
            self.root / 'sleepfm/checkpoints/model_diagnosis/best.pth',
            weights_only=True,
            map_location='cpu',
        )
        self.assertEqual(len(state), 35)
        self.assertTrue(all(name.startswith('module.') for name in state))
        native.load_state_dict(
            {name[7:]: value for name, value in state.items()}, strict=True
        )
        optimizer = torch.optim.AdamW(native.parameters(), lr=0.001)
        loss_source = self.root / 'sleepfm/pipeline/finetune_diagnosis_coxph.py'
        self.assertEqual(
            hashlib.sha256(loss_source.read_bytes()).hexdigest(),
            '810fd6308f3834a995e3c10d16397825d1039aba8ac1f74c026d3690013dadf9',
        )
        functions = [
            node
            for node in ast.parse(loss_source.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == 'cox_ph_loss'
        ]
        self.assertEqual(len(functions), 1)
        scope = {'torch': torch}
        exec(
            compile(
                ast.Module(body=functions, type_ignores=[]), str(loss_source), 'exec'
            ),
            scope,
        )
        tables = {
            kind: csv_rows(paths[name])
            for kind, name in [
                ('event_times', 'event_times_path'),
                ('is_event', 'is_event_path'),
                ('demographics', 'demographics_path'),
            ]
        }
        records = [row for row in manifest['records'] if row['split'] == 'train']
        order = torch.randperm(6, generator=torch.Generator().manual_seed(43)).tolist()
        losses = []
        for offset in [0, 2]:
            selected = [records[index] for index in order[offset : offset + 2]]
            x, mask, demo, times, events = raw_batch(
                self.work / 'data', selected, manifest['schema'], tables
            )
            optimizer.zero_grad(set_to_none=True)
            risks = native(x, mask, demo)
            loss = scope['cox_ph_loss'](risks, times, events)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertEqual(losses, saved['training_losses'])
        for name, value in native.state_dict().items():
            torch.testing.assert_close(
                value,
                saved['state_dict']['model.backend.' + name],
                rtol=0,
                atol=0,
                msg=name,
            )
        assert_payload_equal(optimizer.state_dict(), saved['optimizer'])
        torch.testing.assert_close(
            torch.get_rng_state(), saved['rng']['torch'], rtol=0, atol=0
        )

        # Independent raw-file replay of held-out predictions and risk sets.
        trained = {
            name[len('model.backend.') :]: value
            for name, value in resumed['state_dict'].items()
            if name.startswith('model.backend.')
        }
        self.assertEqual(len(trained), 35)
        native.load_state_dict(trained, strict=True)
        native.eval()
        records = [row for row in manifest['records'] if row['split'] == 'validation']
        x, mask, demo, times, events = raw_batch(
            self.work / 'data', records, manifest['schema'], tables
        )
        with torch.no_grad():
            risks = native(x, mask, demo)
            per_endpoint = []
            for column in range(1065):
                values = []
                for row in torch.where(events[:, column])[0]:
                    at_risk = times[:, column] >= times[row, column]
                    values.append(
                        risks[at_risk, column].logsumexp(0) - risks[row, column]
                    )
                per_endpoint.append(
                    torch.stack(values).mean() if values else risks[:, column].sum() * 0
                )
            loss = torch.stack(per_endpoint).mean()
        torch.testing.assert_close(
            risks, resumed['evaluation']['log_risks'], rtol=0, atol=0
        )
        self.assertAlmostEqual(loss.item(), resumed['evaluation']['loss'], places=6)
        self.assertEqual(
            hashlib.sha256(
                (
                    self.root / 'sleepfm/checkpoints/model_diagnosis/best.pth'
                ).read_bytes()
            ).hexdigest(),
            self.example.SOURCE_FILES['sleepfm/checkpoints/model_diagnosis/best.pth'],
        )

    def test_author_table_schema_with_explicit_synthetic_embeddings_and_one_patient_rejection(
        self,
    ):
        paths, manifest, _ = self.create(
            self.work / 'author_table_fixture', counts=(1, 0, 0)
        )
        demo = self.root / 'notebooks/demo_data'
        with (demo / 'demo_age_gender.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        study = rows[0]['Study ID']
        manifest['records'][0]['study_id'] = study
        manifest['records'][0]['subject_id'] = 'author_demo_SYNTHETIC_ONLY'
        manifest['schema']['demographic_names'] = [
            name for name in rows[0] if name != 'Study ID'
        ]
        manifest['schema']['preprocessing_id'] = (
            'synthetic_fixture_NOT_author_EDF_features'
        )
        manifest['schema']['demographic_encoding_id'] = (
            'author_demo_SYNTHETIC_values_as_supplied_no_codec_inference'
        )
        save_manifest(paths, manifest)
        paths['event_times_path'] = demo / 'time_to_event.csv'
        paths['is_event_path'] = demo / 'is_event.csv'
        paths['demographics_path'] = demo / 'demo_age_gender.csv'
        dataset = SleepFMSurvivalDataset(**paths)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.endpoint_names, tuple(self.endpoint_names))
        model, _ = self.example.build_diagnosis(self.root, dataset.schema)
        objective = SleepFMSurvivalObjective(model, list(dataset.endpoint_names)).eval()
        batch = collate_sleepfm_survival([dataset[0]])
        result = objective(
            batch['embeddings'],
            batch['demographics'],
            batch['event_times'],
            batch['is_event'],
            batch['padding_mask'],
        )
        self.assertEqual(result['log_risks'].shape, (1, 1065))
        self.assertTrue(torch.isfinite(result['log_risks']).all())
        self.assertEqual(result['loss'].item(), 0)
        with self.assertRaisesRegex(ValueError, 'two distinct patients'):
            self.example.batch_ranges(len(dataset), 2)
        bad = dataset.schema
        bad['upstream_checkpoint_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'matching original upstream'):
            self.example.build_diagnosis(self.root, bad)


if __name__ == '__main__':
    unittest.main()
