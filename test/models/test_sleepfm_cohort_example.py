"""Independent file/controller fixtures, not native neural-equivalence evidence."""

import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from torcheeg.datasets import SleepFMSurvivalDataset
from torcheeg.models import SleepFMDiagnosis, SleepFMSurvivalObjective
from test.datasets._sleepfm_cohort import create_cohort, save_manifest
from test.models.test_sleepfm_survival import DiagnosisProtocol


def load_workflow():
    path = (
        Path(__file__).resolve().parents[2]
        / 'examples/examples_sleepfm_survival_training.py'
    )
    spec = importlib.util.spec_from_file_location(
        'sleepfm_cohort_example_protocol', path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_payload_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError('Checkpoint keys differ.')
        for key in a:
            assert_payload_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise AssertionError('Checkpoint sequence lengths differ.')
        for left, right in zip(a, b):
            assert_payload_equal(left, right)
    elif a != b:
        raise AssertionError('Checkpoint values differ.')


class TestSleepFMCohortExample(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        self.paths, self.manifest, _ = create_cohort(self.work / 'data')
        self.example = load_workflow()
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())

    @staticmethod
    def protocol_factory(root, schema):
        # Only the controller is under test here; the complete official model
        # and actual CLI are verified in the separately gated native tests.
        model = SleepFMDiagnosis(
            DiagnosisProtocol(), schema['modality_names'], schema['demographic_names']
        )
        return model, dict(independent_protocol_fixture_only=True)

    def args(self, output, steps, resume=None):
        args = [
            '--backend-root',
            'unused_protocol_fixture',
            '--steps',
            str(steps),
            '--output-path',
            str(output),
        ]
        for name, value in self.paths.items():
            args += ['--' + name.replace('_', '-'), str(value)]
        if resume is not None:
            args += ['--resume', str(resume)]
        return args

    def run_example(self, output, steps, resume=None):
        with patch.object(self.example, 'build_diagnosis', self.protocol_factory):
            with patch('sys.stdout', new=io.StringIO()):
                return self.example.main(self.args(output, steps, resume))

    def test_complete_controller_resume_epoch_cursor_sampler_and_scheduler(self):
        step2 = self.work / 'step2.pt'
        first = self.run_example(step2, 2)
        self.assertEqual(first['cursor']['epoch'], 0)
        self.assertEqual(first['cursor']['offset'], 4)
        resumed = self.run_example(self.work / 'resumed.pt', 3, step2)
        continuous = self.run_example(self.work / 'continuous.pt', 3)
        assert_payload_equal(resumed, continuous)
        self.assertEqual(resumed['cursor']['epoch'], 1)
        self.assertEqual(resumed['cursor']['order'].numel(), 0)
        self.assertEqual(
            resumed['scheduler']['last_epoch'], first['scheduler']['last_epoch'] + 1
        )
        expected_order = torch.randperm(
            6, generator=torch.Generator().manual_seed(43)
        ).tolist()
        self.assertEqual(
            [s for batch in resumed['training_batches'] for s in batch],
            [f'{i:04d}' for i in expected_order],
        )
        self.assertEqual(resumed['evaluation']['study_id'], ['0006', '0007', '0008'])
        self.assertEqual(
            json.loads(resumed['protocol_json'])['cohort']['schema']['data_kind'],
            'synthetic',
        )

    def test_batch_partitions_never_drop_or_duplicate_the_last_patient(self):
        for count in [2, 3, 5, 6, 7, 9]:
            for size in [2, 3, 8]:
                ranges = self.example.batch_ranges(count, size)
                self.assertEqual(
                    [i for start, end in ranges for i in range(start, end)],
                    list(range(count)),
                )
                self.assertTrue(
                    all(2 <= end - start <= size + 1 for start, end in ranges)
                )
        for count, size in [(1, 2), (0, 2), (3, 1), (True, 2), (3, 2.0)]:
            with self.assertRaises(ValueError):
                self.example.batch_ranges(count, size)

    def test_validation_restores_training_mode_rng_and_states_even_after_error(self):
        dataset = SleepFMSurvivalDataset(**self.paths, split='validation')
        model, _ = self.protocol_factory(None, dataset.schema)
        objective = SleepFMSurvivalObjective(
            model, list(dataset.endpoint_names)
        ).train()
        state = copy.deepcopy(objective.state_dict())
        rng = torch.get_rng_state()
        original = self.example._forward

        def noisy(net, batch):
            torch.rand(7)
            return original(net, batch)

        with patch.object(self.example, '_forward', noisy):
            result = self.example.evaluate(objective, dataset, 2)
        self.assertEqual(result['study_id'], ['0006', '0007', '0008'])
        self.assertTrue(objective.training)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        assert_payload_equal(objective.state_dict(), state)

        def failed(net, batch):
            torch.rand(7)
            raise ValueError('independent evaluation failure')

        with patch.object(self.example, '_forward', failed):
            with self.assertRaisesRegex(ValueError, 'evaluation failure'):
                self.example.evaluate(objective, dataset, 2)
        self.assertTrue(objective.training)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_resume_rejects_changed_inputs_protocol_counters_and_permutations(self):
        original = self.work / 'step2.pt'
        saved = self.run_example(original, 2)
        for change in [
            lambda p: p.__setitem__('completed_steps', True),
            lambda p: p['cursor'].__setitem__('epoch', 20),
            lambda p: p['cursor'].__setitem__('offset', 1),
            lambda p: p['cursor'].__setitem__(
                'order', torch.zeros(6, dtype=torch.int64)
            ),
            lambda p: p.__setitem__('training_losses', [1.0]),
        ]:
            payload = copy.deepcopy(saved)
            change(payload)
            corrupt = self.work / ('corrupt_' + str(id(change)) + '.pt')
            torch.save(payload, corrupt)
            output = self.work / ('rejected_' + str(id(change)) + '.pt')
            with self.assertRaises(ValueError):
                self.run_example(output, 3, corrupt)
            self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, 'increase'):
            self.run_example(self.work / 'same_steps.pt', 2, original)
        self.manifest['schema']['preprocessing_id'] = 'changed_protocol'
        save_manifest(self.paths, self.manifest)
        with self.assertRaisesRegex(ValueError, 'protocol differs'):
            self.run_example(self.work / 'changed.pt', 3, original)

    def test_existing_outputs_and_exclusive_publish_are_not_overwritten(self):
        path = self.work / 'keep.pt'
        path.write_bytes(b'keep original')
        with patch('sys.stderr', new=io.StringIO()):
            with self.assertRaises(SystemExit):
                self.run_example(path, 1)
        self.assertEqual(path.read_bytes(), b'keep original')
        with self.assertRaises(FileExistsError):
            self.example.save_checkpoint(path, {'new': torch.ones(1)})
        self.assertEqual(path.read_bytes(), b'keep original')
        self.assertEqual(list(self.work.glob('*.tmp')), [])

    def test_single_subject_training_and_reserved_test_outcome_selection(self):
        for record in self.manifest['records'][1:6]:
            record['split'] = 'test'
        save_manifest(self.paths, self.manifest)
        with self.assertRaisesRegex(ValueError, 'two distinct patients'):
            self.run_example(self.work / 'one.pt', 1)
        self.assertFalse((self.work / 'one.pt').exists())


if __name__ == '__main__':
    unittest.main()
