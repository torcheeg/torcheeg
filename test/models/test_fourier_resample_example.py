from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import unittest


class TestFourierResampleExample(unittest.TestCase):
    def test_numerical_example_runs_without_data_or_network(self):
        example = Path(__file__).resolve().parents[2] / 'examples' / 'examples_fourier_resample.py'
        output = io.StringIO()
        with redirect_stdout(output):
            runpy.run_path(str(example), run_name='__main__')
        result = json.loads(output.getvalue().strip().splitlines()[-1])
        self.assertEqual((result['source_rate_hz'], result['target_rate_hz']), (128, 200))
        self.assertEqual(result['window_seconds'], 4)
        self.assertEqual(result['source_shape'], [2, 3, 512])
        self.assertEqual(result['target_shape'], [2, 3, 800])
        self.assertLess(result['scipy_max_error'], 1e-12)
        self.assertEqual(result['legacy_max_error'], 0)
        self.assertLess(result['patch_nyquist_max_error'], 1e-12)
        self.assertGreater(result['input_gradient_norm'], 0)
        self.assertGreater(result['whole_vs_patch_ramp_difference'], .01)


if __name__ == '__main__':
    unittest.main()
