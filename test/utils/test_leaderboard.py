from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('leaderboard_validator', ROOT / 'docs/validate_leaderboard.py')
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)
DATA_PATH = ROOT / 'docs/source/_static/leaderboard/results.json'


class TestLeaderboardData(unittest.TestCase):

    def setUp(self):
        self.data = VALIDATOR.load(DATA_PATH)

    def test_current_records_have_real_protocols_and_immutable_examples(self):
        self.assertTrue(self.data['results'])
        for result in self.data['results']:
            self.assertIn(result['protocol'], self.data['protocols'])
            self.assertTrue(result['implementation']['example'].startswith('examples/'))
            self.assertTrue(result['evidence']['kind'])

    def test_duplicate_ids_and_unknown_protocols_are_rejected(self):
        duplicate = deepcopy(self.data)
        duplicate['results'].append(deepcopy(duplicate['results'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate result'):
            VALIDATOR.validate(duplicate)
        self.data['results'][0]['protocol'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'unknown protocol'):
            VALIDATOR.validate(self.data)

    def test_subject_overlap_is_rejected(self):
        protocol = self.data['protocols'][self.data['results'][0]['protocol']]
        protocol['split']['test_subjects'] = protocol['split']['train_subjects'][:]
        with self.assertRaisesRegex(ValueError, 'subjects overlap'):
            VALIDATOR.validate(self.data)

    def test_invalid_metrics_are_rejected(self):
        for value in (float('nan'), float('inf'), -0.1, 1.1, True):
            with self.subTest(value=value):
                data = deepcopy(self.data)
                data['results'][0]['metric']['value'] = value
                with self.assertRaises(ValueError):
                    VALIDATOR.validate(data)

    def test_fractional_sampling_rates_and_reported_deviations(self):
        protocol = self.data['protocols'][self.data['results'][0]['protocol']]
        protocol['sampling_rate'] = 125.5
        self.data['results'][0]['metric']['std'] = 0.01
        VALIDATOR.validate(self.data)
        self.data['results'][0]['metric']['std'] = -0.01
        with self.assertRaisesRegex(ValueError, 'deviation'):
            VALIDATOR.validate(self.data)

    def test_example_path_escape_and_incomplete_commit_are_rejected(self):
        for field, value in (('example', 'examples/../../outside.py'),
                             ('example', '/examples/model.py'), ('commit', 'main'), ('sha256', None)):
            with self.subTest(field=field, value=value):
                data = deepcopy(self.data)
                data['results'][0]['implementation'][field] = value
                with self.assertRaises(ValueError):
                    VALIDATOR.validate(data)

    def test_published_results_require_citations_and_allow_unreported_parameters(self):
        result = deepcopy(self.data['results'][0])
        self.data['results'] = [result]
        result['evidence'] = {'kind': 'published'}
        with self.assertRaisesRegex(ValueError, 'citation'):
            VALIDATOR.validate(self.data)
        result['evidence']['citation_url'] = 'https://example.org/published-study'
        protocol = self.data['protocols'][result['protocol']]
        for field in ('epochs', 'seed', 'train_windows', 'test_windows', 'sampling_rate'):
            protocol[field] = None
        protocol.pop('train_class_counts', None)
        protocol.pop('test_class_counts', None)
        protocol['split'] = {'type': 'fixed_subject_holdout', 'train_subjects': None, 'test_subjects': None}
        self.assertIs(VALIDATOR.validate(self.data), self.data)

    def test_measured_runs_require_reported_counts_and_environment(self):
        self.data['results'][0]['environment'] = {}
        with self.assertRaisesRegex(ValueError, 'environment'):
            VALIDATOR.validate(self.data)

    def test_zero_shot_protocol_can_report_zero_training_epochs(self):
        protocol = self.data['protocols'][self.data['results'][0]['protocol']]
        protocol['epochs'] = 0
        protocol['train_windows'] = 0
        protocol.pop('train_class_counts', None)
        protocol['split'] = {'type': 'zero_shot'}
        VALIDATOR.validate(self.data)

    def test_class_counts_must_match_recorded_window_totals(self):
        protocol = self.data['protocols'][self.data['results'][0]['protocol']]
        protocol['test_class_counts']['background'] += 1
        with self.assertRaisesRegex(ValueError, 'class counts do not match'):
            VALIDATOR.validate(self.data)

    def test_duplicate_json_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.json'
            path.write_text('{"schema_version":1,"schema_version":2}')
            with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
                VALIDATOR.load(path)


if __name__ == '__main__':
    unittest.main()
