"""Optional-backend boundaries and full real pretrained YASA classifier parity."""

import importlib.util
from itertools import product
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mne
import numpy as np

from torcheeg.models import YASASleepStaging
from torcheeg.models.traditional.yasa import _WEIGHT_HASHES


class TestYASABoundaries(unittest.TestCase):
    def test_configuration_and_demographic_copy(self):
        for kwargs in [
            dict(eeg_name=''),
            dict(eeg_name=None),
            dict(eeg_name='EEG', eog_name='EEG'),
            dict(eeg_name='EEG', metadata={'age': 30}),
            dict(eeg_name='EEG', metadata={'age': -1, 'male': 1}),
            dict(eeg_name='EEG', metadata={'age': 30, 'male': 2}),
        ]:
            with self.assertRaises(ValueError):
                YASASleepStaging(**kwargs)
        metadata = dict(age=30, male=True)
        model = YASASleepStaging('EEG', metadata=metadata)
        metadata.clear()
        self.assertEqual(model.metadata, dict(age=30.0, male=1))
        self.assertEqual(YASASleepStaging('EEG', metadata={}).metadata, None)

    def test_clear_python_and_optional_dependency_errors(self):
        model = YASASleepStaging('EEG')
        with patch('torcheeg.models.traditional.yasa.sys.version_info', (3, 10)):
            with self.assertRaisesRegex(ImportError, '3.11'):
                model.predict(None)
        with patch('torcheeg.models.traditional.yasa.sys.version_info', (3, 11)), patch(
            'torcheeg.models.traditional.yasa.importlib.import_module',
            side_effect=ModuleNotFoundError,
        ):
            with self.assertRaisesRegex(ImportError, 'optional'):
                model.predict(None)
        with patch('torcheeg.models.traditional.yasa.sys.version_info', (3, 11)), patch(
            'torcheeg.models.traditional.yasa.importlib.import_module',
            return_value=SimpleNamespace(__version__='0.6.5'),
        ):
            with self.assertRaisesRegex(RuntimeError, '0.8.1'):
                model.predict(None)

    def test_only_pinned_official_classifier_bytes_are_accepted(self):
        model = YASASleepStaging('EEG')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'classifiers').mkdir()
            backend = SimpleNamespace(__file__=str(root / '__init__.py'))
            with self.assertRaisesRegex(RuntimeError, 'missing'):
                model._classifier(backend)
            (root / 'classifiers/clf_eeg_lgb_0.5.0.joblib').write_bytes(
                b'not a trusted model'
            )
            with self.assertRaisesRegex(RuntimeError, 'checksum'):
                model._classifier(backend)
        self.assertEqual(len(_WEIGHT_HASHES), 8)


@unittest.skipUnless(
    importlib.util.find_spec('yasa'), 'Optional full YASA 0.8.1 backend not installed.'
)
class TestYASACompleteBackend(unittest.TestCase):
    @staticmethod
    def raw(sfreq=100, seconds=300):
        rng = np.random.default_rng(42)
        times = np.arange(int(sfreq * seconds)) / sfreq
        values = np.stack(
            [
                20e-6 * np.sin(2 * np.pi * 10 * times),
                15e-6 * np.sin(2 * np.pi * 0.7 * times),
                5e-6 * np.sin(2 * np.pi * 22 * times),
            ]
        )
        values += rng.normal(size=values.shape) * 5e-6
        return mne.io.RawArray(
            values,
            mne.create_info(['EEG', 'EOG', 'EMG'], sfreq, ['eeg', 'eog', 'emg']),
            verbose=False,
        )

    def test_all_eight_complete_pretrained_feature_classifier_workflows(self):
        import yasa

        raw = self.raw()
        original = raw.get_data().copy()
        for eog, emg, demographics in product([False, True], repeat=3):
            with self.subTest(eog=eog, emg=emg, demographics=demographics):
                config = dict(
                    eog_name='EOG' if eog else None,
                    emg_name='EMG' if emg else None,
                    metadata=dict(age=30, male=True) if demographics else None,
                )
                native = yasa.SleepStaging(raw, 'EEG', **config).predict()
                actual = YASASleepStaging('EEG', **config).predict(raw)
                expected = native.proba.loc[
                    :, ['WAKE', 'N1', 'N2', 'N3', 'REM']
                ].to_numpy()
                np.testing.assert_array_equal(actual['probabilities'], expected)
                lookup = dict(WAKE=0, N1=1, N2=2, N3=3, REM=4)
                np.testing.assert_array_equal(
                    actual['labels'], [lookup[str(value)] for value in native.hypno]
                )
                np.testing.assert_array_equal(actual['onsets'], np.arange(10) * 30.0)
                self.assertEqual(actual['stage_names'], ('W', 'N1', 'N2', 'N3', 'REM'))
                self.assertEqual(
                    actual['metadata']['classifier_sha256'],
                    _WEIGHT_HASHES[actual['metadata']['classifier_file']],
                )
        np.testing.assert_array_equal(raw.get_data(), original)
        self.assertEqual(raw.ch_names, ['EEG', 'EOG', 'EMG'])
        self.assertEqual(raw.info['sfreq'], 100)

    def test_downsampling_tail_and_cropped_recording_times(self):
        import yasa

        raw = self.raw(sfreq=128, seconds=361)
        raw.crop(tmin=60)
        original = raw.get_data().copy()
        native = yasa.SleepStaging(raw, 'EEG').predict()
        actual = YASASleepStaging('EEG').predict(raw)
        np.testing.assert_array_equal(
            actual['probabilities'],
            native.proba.loc[:, ['WAKE', 'N1', 'N2', 'N3', 'REM']].to_numpy(),
        )
        np.testing.assert_array_equal(
            actual['recording_onsets'], actual['onsets'] + 60.0
        )
        self.assertEqual(actual['metadata']['first_sample'], 60 * 128)
        self.assertAlmostEqual(actual['metadata']['incomplete_tail_seconds'], 1.0)
        self.assertEqual(actual['metadata']['backend_sampling_rate'], 100.0)
        np.testing.assert_array_equal(raw.get_data(), original)
        self.assertEqual(raw.info['sfreq'], 128)

    def test_each_call_extracts_fresh_features_instead_of_caching_raw_identity(self):
        import yasa

        raw = self.raw()
        model = YASASleepStaging('EEG')
        with patch.object(
            yasa, 'SleepStaging', wraps=yasa.SleepStaging
        ) as complete_backend:
            model.predict(raw)
            raw._data[0] *= 3
            actual = model.predict(raw)
            self.assertEqual(complete_backend.call_count, 2)
        native = yasa.SleepStaging(raw, 'EEG').predict()
        np.testing.assert_array_equal(
            actual['probabilities'],
            native.proba.loc[:, ['WAKE', 'N1', 'N2', 'N3', 'REM']].to_numpy(),
        )

    def test_raw_channels_units_sampling_duration_and_finite_boundaries(self):
        model = YASASleepStaging('EEG')
        with self.assertRaises(TypeError):
            model.predict(np.zeros((1, 30000)))
        with self.assertRaises(ValueError):
            YASASleepStaging('missing').predict(self.raw())
        with self.assertRaises(ValueError):
            model.predict(self.raw(sfreq=80))
        with self.assertRaises(ValueError):
            model.predict(self.raw(seconds=29))
        raw = self.raw()
        raw._data[0, 0] = np.nan
        with self.assertRaises(ValueError):
            model.predict(raw)
        raw = self.raw()
        raw.set_channel_types({'EEG': 'misc'})
        with self.assertRaises(ValueError):
            model.predict(raw)


if __name__ == '__main__':
    unittest.main()
