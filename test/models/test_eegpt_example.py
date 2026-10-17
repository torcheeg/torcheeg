import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from moabb.datasets.fake import FakeDataset
from moabb.paradigms import LeftRightImagery


class TestEEGPTExample(unittest.TestCase):
    def test_eegpt_training_cli_without_network(self):
        source = FakeDataset(event_list=['left_hand', 'right_hand'], n_sessions=1,
            n_runs=1, n_subjects=2, seed=42, sfreq=250, duration=40, n_events=4)
        paradigm = LeftRightImagery(channels=['C3', 'Cz', 'C4'], fmin=4, fmax=38,
                                    tmin=0, tmax=4, resample=256)
        expected = {subject: len(paradigm.get_data(source, subjects=[subject])[0])
                    for subject in [1, 2]}
        self.assertTrue(all(count > 0 for count in expected.values()))
        example = Path(__file__).resolve().parents[2] / 'examples' / 'examples_bciciv2b_eegpt.py'
        with tempfile.TemporaryDirectory() as folder:
            arguments = [str(example), '--download-path', str(Path(folder) / 'raw'),
                '--cache-path', str(Path(folder) / 'cache'), '--epochs', '1',
                '--batch-size', '2']
            output = io.StringIO()
            with patch('moabb.datasets.BNCI2014_004', return_value=source), \
                    patch.object(FakeDataset, 'download', return_value=None) as download, \
                    patch('sys.argv', arguments), redirect_stdout(output):
                runpy.run_path(str(example), run_name='__main__')
            download.assert_called_once()
            result = ast.literal_eval(output.getvalue().strip().splitlines()[-1])
            self.assertFalse(result['checkpoint_loaded'])
            self.assertFalse(result['encoder_frozen'])
            self.assertEqual(result['held_out_subject'], 2)
            self.assertEqual(result['train_windows'], expected[1])
            self.assertEqual(result['test_windows'], expected[2])
            self.assertGreaterEqual(result['metrics']['test_accuracy'], 0)
            self.assertLessEqual(result['metrics']['test_accuracy'], 1)


if __name__ == '__main__':
    unittest.main()
