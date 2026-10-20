"""Functional workflow tests use independent protocol fixtures, not author replicas."""

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from torcheeg.datasets import MentalArithmeticDataset, SleepEDFxDataset
from torcheeg.models import SleepFM, SleepFMStaging, SleepFMDiagnosis
from test.models.test_sleepfm import UpstreamFixture, StagingFixture, DiagnosisFixture


EXAMPLES = Path(__file__).resolve().parents[2] / 'examples'


def load_example(name):
    spec = importlib.util.spec_from_file_location(
        'sleepfm_workflow_' + name, EXAMPLES / (name + '.py')
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eegmat_record(record, sampling_rate, **kwargs):
    if sampling_rate != 128:
        raise ValueError('Require128Hz.')
    size = 1280 if record[1] == 0 else 640
    return dict(
        eeg_data=np.random.RandomState(42 + record[1]).normal(size=(19, size)),
        label=record[1],
        record=record,
    )


def sleep_record(record, sfreq, **kwargs):
    if sfreq != 128:
        raise ValueError('Require128Hz.')
    return dict(
        record=record,
        epochs_data=np.random.RandomState(12).normal(scale=1e-5, size=(10, 2, 3840)),
        epochs_label=[
            'Sleep stage W',
            'Sleep stage 1',
            'Sleep stage 2',
            'Sleep stage 3',
            'Sleep stage R',
        ]
        * 2,
        epochs_onsets=np.arange(10) * 30.0,
    )


def diagnosis_metadata():
    return dict(
        modality_names=['BAS', 'RESP'],
        demographic_names=['encoded_age', 'encoded_gender'],
        record_ids=['fixture0', 'fixture1'],
        preprocessing='independent synthetic latent interface fixture',
        demographic_encoding='explicit synthetic encoded columns',
        synthetic=True,
        feature_encoder=dict(
            revision='2bcbae04c3592f61352addb7ac3d4193f0a3ca25',
            checkpoint_sha256='ffc9fc10233ebc4d0aae71abce87070db51b8bac6e4b16a5c1b4401a5f73f799',
            sampling_rate=128,
            patch_samples=640,
            representation='contextual_5s_tokens',
        ),
    )


class TestSleepFMExamples(unittest.TestCase):
    def test_eegmat_complete_workflow_and_frozen_backbone(self):
        module = load_example('examples_eegmat_sleepfm')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            for subject in range(2):
                for condition in [1, 2]:
                    (source / f'Subject{subject:02d}_{condition}.edf').write_bytes(
                        b'fixture only'
                    )
            native = UpstreamFixture()
            state = {k: v.clone() for k, v in native.state_dict().items()}
            checkpoint = root / 'model.pt'
            output = root / 'probe.pt'
            torch.save(
                {'state_dict': {'module.' + k: v for k, v in state.items()}}, checkpoint
            )

            def factory(backend_root, channel_names):
                return UpstreamFixture(), {'protocol_fixture': True}

            args = [
                'example',
                '--root-path',
                str(source),
                '--backend-root',
                str(root),
                '--checkpoint',
                str(checkpoint),
                '--output-path',
                str(output),
                '--cache-path',
                str(root / 'cache'),
                '--epochs',
                '1',
                '--batch-size',
                '2',
            ]
            with (
                patch.object(module, 'build_sleepfm_backend', side_effect=factory),
                patch.object(
                    MentalArithmeticDataset, 'read_record', side_effect=eegmat_record
                ),
                patch('sys.argv', args),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                module.main()
            saved = torch.load(output, weights_only=True)
            meta = json.loads(saved['metadata_json'])
            self.assertEqual(
                (meta['train_windows'], meta['test_windows'], meta['updates']),
                (3, 3, 2),
            )
            self.assertEqual(meta['held_out_subject'], 'Subject01')
            self.assertEqual(meta['sampling_rate'], 128)
            self.assertEqual(meta['window_samples'], 640)
            for k, v in state.items():
                torch.testing.assert_close(
                    saved['state_dict']['backbone.backend.' + k], v, rtol=0, atol=0
                )
            self.assertEqual(saved['test_logits'].shape, (3, 2))

    def test_staging_subject_split_training_export_and_labels(self):
        module = load_example('examples_sleepedfx_sleepfm_staging')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            cassette = source / 'sleep-cassette'
            cassette.mkdir(parents=True)
            for subject in ['SC4001E0', 'SC4011E0']:
                for kind in ['PSG', 'Hypnogram']:
                    (cassette / f'{subject}-{kind}.edf').write_bytes(b'fixture only')
            backend = root / 'author'
            checkpoint = backend / 'sleepfm/checkpoints'
            for name, file in [
                ('model_base', 'best.pt'),
                ('model_sleep_staging', 'best.pth'),
            ]:
                folder = checkpoint / name
                folder.mkdir(parents=True)
                (folder / file).write_bytes(b'fixture provenance')
            base = UpstreamFixture()
            stage = StagingFixture()
            base_state = {k: v.clone() for k, v in base.state_dict().items()}
            stage_state = {k: v.clone() for k, v in stage.state_dict().items()}

            def factory(backend_root):
                encoder = UpstreamFixture()
                encoder.load_state_dict(base_state)
                head = StagingFixture()
                head.load_state_dict(stage_state)
                return SleepFM(encoder, module.CHANNELS), SleepFMStaging(head, ['BAS'])

            output = root / 'stage.pt'
            args = [
                'example',
                '--root-path',
                str(source),
                '--backend-root',
                str(backend),
                '--output-path',
                str(output),
                '--cache-path',
                str(root / 'cache'),
                '--epochs',
                '1',
                '--batch-size',
                '2',
                '--max-windows',
                '4',
            ]
            with (
                patch.object(module, 'build_backends', side_effect=factory),
                patch.object(SleepEDFxDataset, 'read_record', side_effect=sleep_record),
                patch('sys.argv', args),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                module.main()
            saved = torch.load(output, weights_only=True)
            meta = json.loads(saved['metadata_json'])
            self.assertEqual(meta['train_subjects'], ['SC400'])
            self.assertEqual(meta['test_subject'], 'SC401')
            self.assertEqual(meta['updates'], 2)
            self.assertEqual(len(meta['train_ids']), 4)
            self.assertTrue(set(meta['train_ids']).isdisjoint(meta['test_ids']))
            self.assertEqual(saved['test_logits'].shape, (4, 5))
            for k, v in base_state.items():
                torch.testing.assert_close(
                    saved['base_state']['backend.' + k], v, rtol=0, atol=0
                )
            for k, v in stage_state.items():
                if not k.startswith('fc.'):
                    torch.testing.assert_close(
                        saved['staging_state']['backend.' + k], v, rtol=0, atol=0
                    )
            self.assertFalse(
                torch.equal(
                    saved['staging_state']['backend.fc.weight'],
                    stage_state['fc.weight'],
                )
            )

    def test_diagnosis_mapping_and_complete_restore(self):
        module = load_example('examples_sleepfm_diagnosis')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'sleepfm/configs'
            config.mkdir(parents=True)
            path = config / 'label_mapping.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=['label_idx', 'phecode', 'phenotype']
                )
                writer.writeheader()
                writer.writerows(
                    dict(
                        label_idx=i,
                        phecode=str(i),
                        phenotype='synthetic label' + str(i),
                    )
                    for i in reversed(range(1065))
                )
            weights = root / 'sleepfm/checkpoints/model_diagnosis'
            weights.mkdir(parents=True)
            (weights / 'best.pth').write_bytes(b'fixture provenance')
            inputs = root / 'input.pt'
            output = root / 'out.pt'
            mask = torch.zeros(2, 2, 4, dtype=torch.bool)
            mask[1, :, -1] = True
            payload = dict(
                features=torch.randn(2, 2, 4, 128),
                padding_mask=mask,
                demographics=torch.tensor([[0.1, 0.0], [0.2, 1.0]]),
                metadata_json=json.dumps(diagnosis_metadata()),
            )
            torch.save(payload, inputs)
            backend = DiagnosisFixture()
            state = {k: v.clone() for k, v in backend.state_dict().items()}

            def factory(backend_root, modalities, demographics):
                native = DiagnosisFixture()
                native.load_state_dict(state)
                return SleepFMDiagnosis(native, modalities, demographics)

            with (
                patch.object(module, 'build_diagnosis', side_effect=factory),
                patch(
                    'sys.argv',
                    [
                        'example',
                        '--input-path',
                        str(inputs),
                        '--backend-root',
                        str(root),
                        '--output-path',
                        str(output),
                    ],
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                module.main()
            saved = torch.load(output, weights_only=True)
            meta = json.loads(saved['metadata_json'])
            self.assertEqual(saved['log_risk'].shape, (2, 1065))
            self.assertFalse(meta['clinical_validation'])
            self.assertTrue(meta['input_metadata']['synthetic'])
            self.assertEqual(
                [int(r['label_idx']) for r in meta['label_mapping']], list(range(1065))
            )
            for k, v in state.items():
                torch.testing.assert_close(
                    saved['state_dict']['backend.' + k], v, rtol=0, atol=0
                )
            path.write_text('label_idx,phecode,phenotype\n0,0,incomplete\n')
            with self.assertRaises(ValueError):
                module.read_labels(path)

    def test_input_contract_and_window_normalization(self):
        base = load_example('examples_eegmat_sleepfm')
        stage = load_example('examples_sleepedfx_sleepfm_staging')
        for module, shape in [(base, (19, 640)), (stage, (2, 3840))]:
            values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
            expected = (
                (values - values.mean(-1, keepdims=True))
                / values.std(-1, keepdims=True)
            ).astype(np.float32)
            np.testing.assert_allclose(
                module.prepare_window(values)
                if module is base
                else module.prepare_epoch(values),
                expected,
                rtol=0,
                atol=1e-6,
            )
            np.testing.assert_array_equal(
                module.prepare_window(np.zeros(shape))
                if module is base
                else module.prepare_epoch(np.zeros(shape)),
                np.zeros(shape),
            )
        module = load_example('examples_sleepfm_diagnosis')
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'input.pt'
            payload = dict(
                features=torch.zeros(2, 2, 4, 128),
                padding_mask=torch.zeros(2, 2, 4, dtype=torch.bool),
                demographics=torch.zeros(2, 2),
            )
            for change in [
                {'synthetic': None},
                {'demographic_encoding': ''},
                {'record_ids': ['same', 'same']},
                {'feature_encoder': {}},
                {'preprocessing': ''},
            ]:
                meta = diagnosis_metadata()
                meta.update(change)
                torch.save(dict(payload, metadata_json=json.dumps(meta)), p)
                with self.assertRaises(ValueError):
                    module.load_input(p)


if __name__ == '__main__':
    unittest.main()
