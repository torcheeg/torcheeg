import io
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pywt

from torcheeg.transforms import CWTSpectrum


class TestCWTRendering(unittest.TestCase):
    def setUp(self):
        self.signal = np.sin(np.arange(64) / 5)[None, :]

    def test_savefig_options_do_not_change_rgba_pixels(self):
        transform = CWTSpectrum(contourf=True, total_scale=8)
        with plt.rc_context({'figure.figsize': (3.2, 2.4), 'figure.dpi': 100,
                             'savefig.dpi': 'figure', 'savefig.bbox': None}):
            expected = transform(eeg=self.signal)['eeg']
        with plt.rc_context({'figure.figsize': (3.2, 2.4), 'figure.dpi': 100,
                             'savefig.dpi': 150, 'savefig.bbox': 'tight'}):
            actual = transform(eeg=self.signal)['eeg']
        self.assertEqual(actual.shape, (1, 240, 320, 4))
        self.assertEqual(actual.dtype, np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_matches_legacy_pixels_with_figure_dpi(self):
        transform = CWTSpectrum(contourf=True, total_scale=8)
        with plt.rc_context({'figure.figsize': (3.2, 2.4), 'figure.dpi': 100,
                             'savefig.dpi': 'figure', 'savefig.bbox': None}):
            coefficients, frequencies = pywt.cwt(self.signal[0], transform.scales,
                transform.wavelet, 1 / transform.sampling_rate)
            fig = plt.figure()
            try:
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                plt.margins(0, 0)
                time = np.arange(0, self.signal.shape[1] / transform.sampling_rate,
                                 1 / transform.sampling_rate)
                plt.contourf(time, frequencies, abs(coefficients))
                with io.BytesIO() as buffer:
                    fig.savefig(buffer, format='raw')
                    expected = np.frombuffer(buffer.getvalue(), dtype=np.uint8).reshape(240, 320, 4)
                actual = transform(eeg=self.signal)['eeg'][0]
                np.testing.assert_array_equal(actual, expected)
            finally:
                plt.close(fig)

    def test_preserves_callers_figures(self):
        caller = plt.figure()
        try:
            before = plt.get_fignums()
            CWTSpectrum(contourf=True, total_scale=8)(eeg=self.signal)
            self.assertEqual(plt.get_fignums(), before)
            self.assertIs(plt.gcf(), caller)
        finally:
            plt.close(caller)

    def test_render_failure_does_not_leave_figures(self):
        before = plt.get_fignums()
        with patch('matplotlib.backends.backend_agg.FigureCanvasAgg.draw',
                   side_effect=RuntimeError('injected render error')):
            with self.assertRaisesRegex(RuntimeError, 'render error'):
                CWTSpectrum(contourf=True, total_scale=8)(eeg=self.signal)
        self.assertEqual(plt.get_fignums(), before)

    def test_figure_size_and_dpi_determine_shape(self):
        with plt.rc_context({'figure.figsize': (2, 1), 'figure.dpi': 80,
                             'savefig.dpi': 120}):
            image = CWTSpectrum(contourf=True, total_scale=8)(eeg=self.signal)['eeg']
        self.assertEqual(image.shape, (1, 80, 160, 4))

    def test_coefficients_match_pywavelets(self):
        transform = CWTSpectrum(total_scale=8)
        expected, _ = pywt.cwt(self.signal[0], transform.scales, transform.wavelet,
                               1 / transform.sampling_rate)
        np.testing.assert_array_equal(transform(eeg=self.signal)['eeg'][0], expected)


if __name__ == '__main__':
    unittest.main()
