import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mne
import numpy as np

from torcheeg import transforms
from torcheeg.datasets import SHHSDataset, SequenceDataset
from torcheeg.datasets.module.sleep_stage_detection.shhs import _scored_epochs


MODULE = 'torcheeg.datasets.module.sleep_stage_detection.shhs'


def write_xml(path, intervals=None, epoch=30):
    intervals = [(0, 30, 0), (30, 30, 1), (60, 30, 5)] if intervals is None else intervals
    root = ET.Element('PSGAnnotation')
    ET.SubElement(root, 'EpochLength').text = str(epoch)
    events = ET.SubElement(root, 'ScoredEvents')
    # A respiratory event is not a sleep-stage annotation.
    event = ET.SubElement(events, 'ScoredEvent')
    for name, value in [('EventType', 'Respiratory|Respiratory'), ('EventConcept', 'Hypopnea|Hypopnea'),
                        ('Start', '3'), ('Duration', '7')]:
        ET.SubElement(event, name).text = value
    for onset, duration, code in intervals:
        event = ET.SubElement(events, 'ScoredEvent')
        for name, value in [('EventType', 'Stages|Stages'), ('EventConcept', f'Fixture|{code}'),
                            ('Start', str(onset)), ('Duration', str(duration))]:
            ET.SubElement(event, name).text = value
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def write_edf(path, seconds=95):
    # A real, anonymous EDF with different channel rates and known calibration.
    names, rates = ['EEG', 'EOG(L)'], [100, 50]
    arrays = [np.rint(20 * np.sin(2 * np.pi * 3 * np.arange(seconds * rate) / rate)).astype('<i2')
              for rate in rates]
    def field(value, size):
        return str(value).encode('ascii').ljust(size)
    header = b''.join(field(value, size) for value, size in [
        ('0', 8), ('X', 80), ('X', 80), ('01.01.00', 8), ('00.00.00', 8),
        (768, 8), ('', 44), (seconds, 8), (1, 8), (2, 4)])
    signal_header = b''.join(field(value, size) for values, size in [
        (names, 16), (['', ''], 80), (['uV', 'uV'], 8), ([-32768] * 2, 8),
        ([32767] * 2, 8), ([-32768] * 2, 8), ([32767] * 2, 8),
        (['', ''], 80), (rates, 8), (['', ''], 32)] for value in values)
    with path.open('wb') as handle:
        handle.write(header + signal_header)
        for second in range(seconds):
            for values, rate in zip(arrays, rates):
                handle.write(values[second * rate:(second + 1) * rate].tobytes())
    return arrays


class TestSHHSDataset(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        folder = self.root / 'polysomnography'
        self.edf = folder / 'edfs/shhs1/shhs1-200001.edf'
        self.xml = folder / 'annotations-events-nsrr/shhs1/shhs1-200001-nsrr.xml'
        self.edf.parent.mkdir(parents=True)
        self.xml.parent.mkdir(parents=True)
        self.arrays = write_edf(self.edf)
        write_xml(self.xml)
        self.record = (str(self.edf), str(self.xml))

    def read(self, **kwargs):
        settings = dict(channels=['EEG', 'EOG(L)'], l_freq=None, h_freq=None)
        settings.update(kwargs)
        return SHHSDataset.read_record(self.record, **settings)

    def test_real_edf_exact_grid_labels_units_and_native_sample_values(self):
        result = self.read()
        self.assertEqual(result['epochs_data'].shape, (3, 2, 3000))
        self.assertEqual(result['epochs_label'], ['Sleep stage W', 'Sleep stage N1', 'Sleep stage R'])
        self.assertEqual(result['epochs_units'], ['V', 'V'])
        np.testing.assert_equal(result['epochs_onsets'], [0, 30, 60])
        data = result['epochs_data'].transpose(1, 0, 2).reshape(2, -1)
        np.testing.assert_allclose(data[0], self.arrays[0][:9000] * 1e-6, rtol=0, atol=1e-18)
        np.testing.assert_allclose(data[1, ::2], self.arrays[1][:4500] * 1e-6, rtol=0, atol=1e-18)

    def test_requested_order_resampling_and_nondefault_epoch_size(self):
        result = self.read(channels=['EOG(L)', 'EEG'], sfreq=137)
        self.assertEqual(result['epochs_data'].shape, (3, 2, 4110))
        self.assertEqual(result['epochs_channels'], ['EOG(L)', 'EEG'])
        np.testing.assert_equal(result['epochs_onsets'], [0, 30, 60])
        with mne.io.read_raw_edf(str(self.edf), preload=True, verbose=False) as raw:
            raw.reorder_channels(['EOG(L)', 'EEG'])
            raw.resample(137, npad='auto', verbose=False)
            reference = raw.get_data()[:, :12330].reshape(2, 3, 4110).transpose(1, 0, 2)
        np.testing.assert_equal(result['epochs_data'], reference)

    def test_stage_three_four_merge_long_intervals_and_unknown_gaps(self):
        write_edf(self.edf, seconds=240)
        write_xml(self.xml, [(0, 30, 0), (30, 60, 3), (90, 30, 4),
                             (120, 30, 6), (150, 30, 9), (180, 60, 5)])
        result = self.read()
        np.testing.assert_equal(result['epochs_onsets'], [0, 30, 60, 90, 180, 210])
        self.assertEqual(result['epochs_label'], ['Sleep stage W'] + ['Sleep stage N3'] * 3
                         + ['Sleep stage R'] * 2)

    def test_crop_preserves_grid_and_unannotated_gaps(self):
        write_xml(self.xml, [(0, 30, 0), (60, 30, 5)])
        result = self.read(start_at=15, end_at=1000)
        np.testing.assert_equal(result['epochs_onsets'], [60])
        np.testing.assert_equal(result['epochs_data'][0, 0], self.arrays[0][6000:9000] * 1e-6)
        with self.assertRaisesRegex(ValueError, 'no complete scored'):
            self.read(start_at=0, end_at=29.9)
        with self.assertRaisesRegex(ValueError, 'no complete scored'):
            self.read(start_at=95)

    def test_incomplete_signal_tail_does_not_get_a_fabricated_epoch(self):
        write_xml(self.xml, [(0, 120, 0)])
        result = self.read()
        self.assertEqual(len(result['epochs_data']), 3)
        np.testing.assert_equal(result['epochs_onsets'], [0, 30, 60])

    def test_fir_filter_matches_mne_and_has_finite_inband_amplitude(self):
        result = self.read(l_freq=0.5, h_freq=20, sfreq=50, channels=['EEG'])
        with mne.io.read_raw_edf(str(self.edf), include=['EEG'], preload=True, verbose=False) as raw:
            raw.filter(0.5, 20, picks='all', method='fir', fir_design='firwin', verbose=False)
            raw.resample(50, npad='auto', verbose=False)
            reference = raw.get_data()[:, :4500].reshape(1, 3, 1500).transpose(1, 0, 2)
        np.testing.assert_equal(result['epochs_data'], reference)
        self.assertTrue(np.isfinite(result['epochs_data']).all())
        # The central 3-Hz signal is 20 microvolts, with EDF integer quantization.
        self.assertAlmostEqual(np.max(result['epochs_data'][1]), 20e-6, delta=1e-6)
        for low, high in [(None, 20), (0.5, None)]:
            self.assertTrue(np.isfinite(self.read(l_freq=low, h_freq=high)['epochs_data']).all())

    def test_invalid_configuration_is_rejected_before_io(self):
        cases = [dict(channels=[]), dict(channels=['EEG', 'EEG']), dict(channels=[0]),
                 dict(sfreq=False), dict(sfreq=0), dict(sfreq=np.inf), dict(sfreq=100.01),
                 dict(l_freq=-1), dict(h_freq=50), dict(l_freq=30, h_freq=20),
                 dict(start_at=-1), dict(start_at=True), dict(end_at=np.nan), dict(end_at=0)]
        with patch(MODULE + '.mne.io.read_raw_edf') as reader:
            for args in cases:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    self.read(**args)
            reader.assert_not_called()
        for visits in [None, [], [True], [3], [1, 1], '1']:
            with self.subTest(visits=visits), patch(MODULE + '.BaseDataset.__init__') as parent:
                with self.assertRaises(ValueError):
                    SHHSDataset(visits=visits)
                parent.assert_not_called()

    def test_invalid_annotations_are_rejected(self):
        cases = [[(0, 30, 10)], [(0, 15, 0)], [(1, 30, 0)], [(0, -30, 0)],
                 [(0, np.nan, 0)], [(0, 60, 0), (30, 30, 1)],
                 [(0, 30, 9)], []]
        for intervals in cases:
            with self.subTest(intervals=intervals):
                write_xml(self.xml, intervals)
                with self.assertRaises(ValueError):
                    self.read()
        write_xml(self.xml, epoch=20)
        with self.assertRaisesRegex(ValueError, '30-second'):
            self.read()
        self.xml.write_text('<CMPStudyConfig><SleepStages/></CMPStudyConfig>')
        with self.assertRaisesRegex(ValueError, 'NSRR'):
            self.read()

    def test_malformed_stage_fields_and_unordered_xml(self):
        write_xml(self.xml, [(60, 30, 5), (0, 30, 0), (30, 30, 1)])
        self.assertEqual([stage for _, stage in _scored_epochs(self.xml)],
                         ['Sleep stage W', 'Sleep stage N1', 'Sleep stage R'])
        for field in ['EventConcept', 'Start', 'Duration']:
            write_xml(self.xml)
            tree = ET.parse(self.xml)
            tree.getroot().findall('./ScoredEvents/ScoredEvent')[1].remove(
                tree.getroot().findall('./ScoredEvents/ScoredEvent')[1].find(field))
            tree.write(self.xml)
            with self.assertRaisesRegex(ValueError, 'Invalid NSRR'):
                self.read()

    def test_missing_channels_and_mismatched_pair_are_explicit(self):
        with self.assertRaisesRegex(ValueError, 'Missing required'):
            self.read(channels=['EEG', 'missing'])
        with patch(MODULE + '.mne.io.read_raw_edf') as reader:
            for record in [(str(self.edf), 'shhs1-200002-nsrr.xml'),
                           ('wrong.edf', str(self.xml))]:
                with self.assertRaisesRegex(ValueError, 'matching NSRR'):
                    SHHSDataset.read_record(record)
            reader.assert_not_called()

    def test_nonfinite_signal_is_rejected_and_raw_is_closed(self):
        raw = mne.io.RawArray(np.full((1, 10000), np.nan),
                              mne.create_info(['EEG'], 100, 'eeg'), verbose=False)
        with patch(MODULE + '.mne.io.read_raw_edf', return_value=raw), \
                patch.object(raw, 'close') as close:
            with self.assertRaisesRegex(ValueError, 'finite'):
                self.read(channels=['EEG'])
            close.assert_called_once()

    def test_native_nonvoltage_units_and_missing_unit_are_not_called_volts(self):
        raw = mne.io.RawArray(np.zeros((2, 10000)),
                              mne.create_info(['EEG', 'SaO2'], 100, ['eeg', 'misc']), verbose=False)
        raw._orig_units = {'EEG': 'µV', 'SaO2': '%'}
        with patch(MODULE + '.mne.io.read_raw_edf', return_value=raw):
            result = self.read(channels=['EEG', 'SaO2'])
        self.assertEqual(result['epochs_units'], ['V', '%'])
        raw = mne.io.RawArray(np.zeros((1, 10000)),
                              mne.create_info(['EEG'], 100, 'eeg'), verbose=False)
        with patch(MODULE + '.mne.io.read_raw_edf', return_value=raw):
            self.assertEqual(self.read(channels=['EEG'])['epochs_units'], [None])

    def test_nonzero_raw_sample_origin_keeps_recording_relative_xml_clock(self):
        time = np.arange(9500) / 100.
        raw = mne.io.RawArray(np.sin(time)[None] * 1e-6,
                              mne.create_info(['EEG'], 100, 'eeg'), first_samp=1000, verbose=False)
        with patch(MODULE + '.mne.io.read_raw_edf', return_value=raw):
            result = self.read(channels=['EEG'], sfreq=50, start_at=30)
        np.testing.assert_equal(result['epochs_onsets'], [30., 60.])
        self.assertEqual(result['epochs_data'].shape, (2, 1, 1500))

    def test_deterministic_discovery_both_visits_and_pair_integrity(self):
        edf2 = self.edf.parent.parent / 'shhs2/shhs2-200001.edf'
        xml2 = self.xml.parent.parent / 'shhs2/shhs2-200001-nsrr.xml'
        edf2.parent.mkdir(); xml2.parent.mkdir()
        edf2.touch(); xml2.touch()
        self.assertEqual(SHHSDataset.set_records(None, self.root), [self.record, (str(edf2), str(xml2))])
        self.assertEqual(SHHSDataset.set_records(None, self.root / 'polysomnography', visits=[2]),
                         [(str(edf2), str(xml2))])
        xml2.unlink()
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            SHHSDataset.set_records(None, self.root)
        with self.assertRaisesRegex(ValueError, 'No SHHS'):
            SHHSDataset.set_records(None, self.root / 'missing')

    def test_discovery_rejects_malformed_files_without_shifting_pairs(self):
        (self.edf.parent / 'wrong.edf').touch()
        with self.assertRaisesRegex(ValueError, 'Unrecognized'):
            SHHSDataset.set_records(None, self.root)

    def test_metadata_groups_person_across_visits_and_applies_hooks(self):
        result = self.read()
        rows = list(SHHSDataset.process_record(self.record, **result,
                    before_trial=lambda x: x * 2, offline_transform=lambda eeg: dict(eeg=eeg + 1)))
        alternate = ('shhs2-200001.edf', 'shhs2-200001-nsrr.xml')
        second = next(SHHSDataset.process_record(alternate, **result))
        for i, row in enumerate(rows):
            info = row['info']
            self.assertEqual(info['subject_id'], second['info']['subject_id'])
            self.assertEqual(info['session_id'], 1)
            self.assertEqual(second['info']['session_id'], 2)
            self.assertNotEqual(info['clip_id'], second['info']['clip_id'])
            self.assertEqual(info['trial_id'], info['record_id'])
            self.assertEqual(info['start_at'], i * 30.)
            self.assertEqual(info['end_at'], (i + 1) * 30.)
            self.assertEqual(json.loads(info['channel_units']), ['V', 'V'])
            self.assertEqual(json.loads(info['channel_names']), ['EEG', 'EOG(L)'])
            np.testing.assert_equal(row['eeg'], result['epochs_data'][i] * 2 + 1)

    def test_corrupt_process_metadata_and_hook_count_are_rejected(self):
        original = self.read()
        for changes in [dict(epochs_data=np.zeros((3, 100))), dict(epochs_label=['bad'] * 3),
                        dict(epochs_onsets=[0, 29, 60]), dict(epochs_onsets=[0, np.nan, 60]),
                        dict(epochs_channels=['EEG']), dict(epochs_channels=['EEG', 'EEG']),
                        dict(epochs_units=[]), dict(before_trial=lambda x: x[:1])]:
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                next(SHHSDataset.process_record(self.record, **dict(original, **changes)))

    def test_real_cache_transforms_sequences_and_reload(self):
        write_edf(self.edf, seconds=180)
        write_xml(self.xml, [(0, 60, 0), (60, 30, 9), (90, 90, 5)])
        settings = dict(root_path=str(self.root), io_path=str(self.root / 'cache'), io_mode='pickle',
                        channels=['EEG', 'EOG(L)'], l_freq=None, h_freq=None, verbose=False,
                        online_transform=transforms.ToTensor(), label_transform=transforms.Select('label'))
        dataset = SHHSDataset(**settings)
        self.assertEqual(len(dataset), 5)
        signal, label = dataset[0]
        self.assertEqual(tuple(signal.shape), (2, 3000))
        self.assertEqual(label, 'Sleep stage W')
        sequence = SequenceDataset(dataset, sequence_length=2, stride=1)
        self.assertEqual(len(sequence), 3)
        self.assertEqual(tuple(sequence[0][0].shape), (2, 2, 3000))
        reloaded = SHHSDataset(**settings)
        np.testing.assert_equal(reloaded[0][0].numpy(), dataset[0][0].numpy())
        copied = copy.copy(dataset)
        self.assertEqual(copied.info['start_at'].tolist(), [0., 30., 90., 120., 150.])
        self.assertIn('SHHSDataset', repr(dataset))


if __name__ == '__main__':
    unittest.main()
