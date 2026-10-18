import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import signal
from scipy.io import savemat
import wfdb

from torcheeg.datasets import P2018Dataset, SequenceDataset
from torcheeg import transforms


MODULE = 'torcheeg.datasets.module.sleep_stage_detection.p2018'


def write_record(root, name='tr03-0001', fs=200, seconds=95, events=None):
    folder = Path(root, name)
    folder.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(fs * seconds)) / fs
    digital = np.stack([np.round(1000 * np.sin(2 * np.pi * 10 * time)),
                        np.round(500 * np.cos(2 * np.pi * 20 * time))], axis=1).astype(np.int16)
    savemat(str(folder / (name + '.mat')), {'val': digital.T}, format='4')
    header = [f'{name} 2 {fs} {len(digital)}']
    for index, (channel, gain, unit) in enumerate([('C3-M2', 100, 'uV'), ('E1-M2', 50, 'mV')]):
        checksum = (int(digital[:, index].sum(dtype=np.int64)) + 32768) % 65536 - 32768
        header.append(f'{name}.mat 16+24 {gain}(0)/{unit} 16 0 {digital[0,index]} {checksum} 0 {channel}')
    (folder / (name + '.hea')).write_text('\n'.join(header) + '\n')
    events = [(0, 'W'), (30, 'N1'), (60, 'R')] if events is None else events
    wfdb.wrann(name, 'arousal', sample=np.array([round(t * fs) for t, _ in events], dtype=np.int64),
               # WFDB reserves NOTE annotations at sample zero for file metadata.
               symbol=['+' if t == 0 else '"' for t, _ in events],
               aux_note=[note for _, note in events], fs=fs, write_dir=str(folder))
    return (str(folder / (name + '.mat')), str(folder / (name + '.arousal'))), digital / np.array([100., 50.])


class TestP2018Dataset(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.record, self.physical = write_record(self.root)

    def read(self, **kwargs):
        options = dict(channels=['C3-M2', 'E1-M2'], l_freq=None, h_freq=None)
        options.update(kwargs)
        return P2018Dataset.read_record(self.record, **options)

    def test_zero_first_stage_never_reinitializes_the_label_origin(self):
        result = self.read()
        self.assertEqual(result['epochs_label'], ['Sleep stage W', 'Sleep stage N1', 'Sleep stage R'])
        np.testing.assert_equal(result['epochs_onsets'], [0., 30., 60.])
        expected = signal.resample(self.physical, int(len(self.physical) / 2), axis=0)
        # Batched and per-channel FFT layouts can differ by binary64 roundoff.
        np.testing.assert_allclose(result['epochs_data'], expected[:9000].reshape(3, 3000, 2).transpose(0, 2, 1),
                                   rtol=1e-12, atol=1e-12)

    def test_header_rate_and_output_rate_determine_signal_and_epoch_lengths(self):
        self.record, physical = write_record(self.root, name='tr03-0002', fs=128)
        for output_rate in (100, 137, 128):
            result = self.read(sfreq=output_rate)
            self.assertEqual(result['epochs_data'].shape, (3, 2, 30 * output_rate))
            expected = (physical if output_rate == 128 else
                        signal.resample(physical, int(len(physical) * output_rate / 128), axis=0))
            for i, chunk in enumerate(result['epochs_data']):
                reference = expected[i * 30 * output_rate:(i + 1) * 30 * output_rate].T
                if output_rate == 128:
                    np.testing.assert_equal(chunk, reference)
                else:
                    np.testing.assert_allclose(chunk, reference, rtol=1e-12, atol=1e-12)
            np.testing.assert_equal(result['epochs_onsets'], [0., 30., 60.])
            self.assertEqual(result['epochs_label'], ['Sleep stage W', 'Sleep stage N1', 'Sleep stage R'])

    def test_fractional_scoring_origin_rounds_forward_on_the_output_sample_grid(self):
        self.record, _ = write_record(self.root, fs=250,
                     events=[(.4, 'W'), (30.4, 'N1'), (60.4, 'R')])
        result = self.read(sfreq=128)
        self.assertEqual(result['epochs_data'].shape, (3, 2, 3840))
        np.testing.assert_equal(result['epochs_onsets'], np.array([52, 3892, 7732]) / 128.)
        self.assertEqual(result['epochs_label'], ['Sleep stage W', 'Sleep stage N1', 'Sleep stage R'])
        self.assertGreaterEqual(result['epochs_onsets'][0], .4)
        self.assertLess(result['epochs_onsets'][0] + 3839 / 128., 30.4)

    def test_arousal_before_sleep_scoring_does_not_define_the_signal_crop(self):
        self.record, physical = write_record(self.root, seconds=125,
                 events=[(3, '(arousal_rera'), (10, 'arousal_rera)'), (30, 'W'), (60, 'N1'), (90, 'R')])
        result = self.read()
        np.testing.assert_equal(result['epochs_onsets'], [30., 60., 90.])
        self.assertEqual(result['epochs_label'], ['Sleep stage W', 'Sleep stage N1', 'Sleep stage R'])
        expected = signal.resample(physical, int(len(physical) / 2), axis=0)
        np.testing.assert_allclose(result['epochs_data'][0], expected[3000:6000].T, rtol=1e-12, atol=1e-12)

    def test_undefined_intervals_and_mid_epoch_changes_are_not_filled_or_bridged(self):
        self.record, _ = write_record(self.root, seconds=125,
                         events=[(0, 'W'), (30, 'undefined'), (60, 'N2'), (90, 'R')])
        result = self.read()
        np.testing.assert_equal(result['epochs_onsets'], [0., 60., 90.])
        self.assertEqual(result['epochs_label'], ['Sleep stage W', 'Sleep stage N2', 'Sleep stage R'])
        self.record, _ = write_record(self.root, events=[(0, 'W'), (45, 'N1')])
        result = self.read()
        np.testing.assert_equal(result['epochs_onsets'], [0., 60.])
        self.assertEqual(result['epochs_label'], ['Sleep stage W', 'Sleep stage N1'])

    def test_repeated_stage_markers_do_not_create_artificial_gaps(self):
        self.record, _ = write_record(self.root, events=[(0, 'W'), (15, 'W'), (30, 'W'), (60, 'R')])
        result = self.read()
        np.testing.assert_equal(result['epochs_onsets'], [0., 30., 60.])
        self.assertEqual(result['epochs_label'], ['Sleep stage W', 'Sleep stage W', 'Sleep stage R'])

    def test_conflicting_and_unknown_stages_and_unscored_only_records_raise(self):
        for events in [[(0, 'W'), (30, 'W'), (30, 'N1')], [(0, 'Q')], [(0, 'undefined')],
                       [(0, '(arousal_rera'), (10, 'arousal_rera)')], [(90, 'W')]]:
            self.record, _ = write_record(self.root, events=events)
            with self.subTest(events=events), self.assertRaises(ValueError):
                self.read()

    def test_reversed_channels_and_native_physical_units_are_preserved(self):
        result = self.read(channels=['E1-M2', 'C3-M2'])
        self.assertEqual(result['epochs_channels'], ['E1-M2', 'C3-M2'])
        self.assertEqual(result['epochs_units'], ['mV', 'uV'])
        expected = signal.resample(self.physical, int(len(self.physical) / 2), axis=0)
        np.testing.assert_allclose(result['epochs_data'][0], expected[:3000, [1, 0]].T, rtol=1e-12, atol=1e-12)

    def test_butterworth_filter_is_finite_and_keeps_in_band_signal_amplitudes(self):
        result = self.read(l_freq=.5, h_freq=30.)
        self.assertTrue(np.isfinite(result['epochs_data']).all())
        # Check measured central-epoch amplitude, excluding edge transients.
        middle = result['epochs_data'][1]
        for channel, frequency in [(0, 10), (1, 20)]:
            time = np.arange(3000) / 100.
            amplitude = abs((middle[channel] * np.exp(-2j * np.pi * frequency * time)).mean()) * 2
            self.assertAlmostEqual(amplitude, 10., delta=.02)
        for low, high in [(None, 30.), (.5, None)]:
            self.assertTrue(np.isfinite(self.read(l_freq=low, h_freq=high)['epochs_data']).all())

    def test_metadata_and_hooks_preserve_offsets_labels_and_unit_provenance(self):
        self.record, _ = write_record(self.root, seconds=125, events=[(30, 'W'), (60, 'N1'), (90, 'R')])
        original = self.read()
        rows = list(P2018Dataset.process_record(self.record, before_trial=lambda x: x * 2,
                     offline_transform=lambda eeg: {'eeg': eeg + 1}, **original))
        self.assertEqual(rows[0]['info']['subject_id'], 'tr03-0001')
        self.assertEqual(rows[0]['info']['record_id'], 'tr03-0001')
        self.assertEqual(rows[0]['info']['trial_id'], 'tr03-0001')
        self.assertEqual(rows[0]['info']['start_at'], 30.)
        self.assertEqual(rows[-1]['info']['end_at'], 120.)
        self.assertEqual(json.loads(rows[0]['info']['channel_names']), ['C3-M2', 'E1-M2'])
        self.assertEqual(json.loads(rows[0]['info']['channel_units']), ['uV', 'mV'])
        np.testing.assert_equal(rows[1]['eeg'], original['epochs_data'][1] * 2 + 1)

    def test_file_discovery_validates_signal_header_and_annotations(self):
        (self.root / 'README.txt').touch()
        (self.root / 'notes').mkdir()
        records = P2018Dataset.set_records(None, str(self.root))
        self.assertEqual(records, [self.record])
        Path(self.record[0]).with_suffix('.hea').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            P2018Dataset.set_records(None, str(self.root))

    def test_record_pair_and_configuration_fail_before_signal_io(self):
        for record in [(self.record[0], str(self.root / 'tr03-0002.arousal')),
                       ('wrong.mat', 'wrong.arousal')]:
            with patch(MODULE + '.wfdb.rdsamp') as reader:
                with self.assertRaises(ValueError):
                    P2018Dataset.read_record(record)
                reader.assert_not_called()
        for kwargs in [dict(sfreq=0), dict(sfreq=True), dict(sfreq=np.inf), dict(sfreq=100.01),
                       dict(l_freq=0), dict(h_freq=50), dict(l_freq=30, h_freq=20),
                       dict(channels=[]), dict(channels=['C3-M2', 'C3-M2'])]:
            with self.subTest(kwargs=kwargs), patch(MODULE + '.BaseDataset.__init__') as parent:
                with self.assertRaises(ValueError):
                    P2018Dataset(root_path='unused', io_path='unused', **kwargs)
                parent.assert_not_called()

    def test_cached_dataset_and_sequences_retain_undefined_gaps(self):
        self.record, _ = write_record(self.root, seconds=155,
               events=[(0, 'W'), (60, 'undefined'), (90, 'N2')])
        dataset = P2018Dataset(root_path=str(self.root), channels=['C3-M2', 'E1-M2'],
                              l_freq=None, h_freq=None, io_path=str(self.root / 'cache'),
                              io_mode='pickle', num_worker=0, verbose=False,
                              label_transform=transforms.Select('label'))
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.info['start_at'].tolist(), [0., 30., 90., 120.])
        sequences = SequenceDataset(dataset, sequence_length=2)
        self.assertEqual(len(sequences), 2)
        self.assertEqual(sequences.get_epoch_indices(0), (0, 1))
        self.assertEqual(sequences.get_epoch_indices(1), (2, 3))
        self.assertEqual(list(sequences[0][1]), ['Sleep stage W', 'Sleep stage W'])
        self.assertEqual(list(sequences[1][1]), ['Sleep stage N2', 'Sleep stage N2'])

    def test_invalid_raw_signals_annotation_clock_and_process_metadata_are_rejected(self):
        annotation = SimpleNamespace(sample=np.array([0]), aux_note=['W'], fs=100)
        with patch(MODULE + '.wfdb.rdann', return_value=annotation):
            with self.assertRaisesRegex(ValueError, 'sampling rates'):
                self.read()
        for values in [np.ones((20,)), np.full((100, 2), np.nan)]:
            with patch(MODULE + '.wfdb.rdsamp', return_value=(values, {'fs': 200})):
                with self.assertRaises(ValueError):
                    self.read()
        original = self.read()
        for changed in [dict(epochs_label=['Sleep stage W']), dict(epochs_onsets=[0, 30, np.nan]),
                        dict(epochs_onsets=[0, 29, 60]), dict(epochs_channels=['other']),
                        dict(epochs_units=['uV']), dict(before_trial=lambda x: x[:1])]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                next(P2018Dataset.process_record(self.record, **dict(original, **changed)))


if __name__ == '__main__':
    unittest.main()
