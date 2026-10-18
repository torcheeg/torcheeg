import copy
import csv
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import h5py
import mne
import numpy as np

from torcheeg import transforms
from torcheeg.datasets import ADAuditoryDataset
from torcheeg.datasets.module.clinical.ad_auditory import AD_AUDITORY_CHANNELS


def write_record(root, subject='sub-01', run=None, permutation=None):
    folder = Path(root) / subject / 'eeg'
    folder.mkdir(parents=True, exist_ok=True)
    identity = subject + '_task-40HzAuditoryEntrainment' + (f'_run-{run}' if run else '')
    path = folder / (identity + '_eeg.set')
    time = np.arange(2500) / 250.
    values = np.stack([(index + 1) * np.sin(2 * np.pi * (4 + index) * time) + index
                       for index in range(19)], axis=1).astype('<f4')
    order = list(range(19)) if permutation is None else permutation
    values[:, order].tofile(path.with_suffix('.fdt'))
    with h5py.File(path, 'w') as file:
        for key, value in [('nbchan', 19), ('srate', 250), ('pnts', 2500), ('trials', 1)]:
            file.create_dataset(key, data=[[float(value)]])
        file.create_dataset('data', data=np.array([ord(char) for char in path.with_suffix('.fdt').name], dtype=np.uint16)[:, None])
        references = file.create_group('#refs#')
        labels = file.create_group('chanlocs').create_dataset('labels', shape=(19, 1), dtype=h5py.ref_dtype)
        for index, channel in enumerate(order):
            dataset = references.create_dataset(str(index), data=np.array([ord(char) for char in AD_AUDITORY_CHANNELS[channel]], dtype=np.uint16)[:, None])
            labels[index, 0] = dataset.ref
    events = [dict(onset=1., duration=2., sample=251, value=2, trial_type='Stimulus'),
              dict(onset=3., duration=1., sample=751, value=1, trial_type='Rest'),
              dict(onset=4., duration=2., sample=1001, value=2, trial_type='Stimulus'),
              dict(onset=6., duration=.5, sample=1501, value=1, trial_type='Rest')]
    write_events(path, events)
    return path, values.T, events


def write_events(path, events):
    target = path.with_name(path.name.replace('_eeg.set', '_events.tsv'))
    with target.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['trial_type', 'onset', 'duration', 'sample', 'value'], delimiter='\t')
        writer.writeheader()
        writer.writerows(events)


def participants(root, entries=(('sub-01', 'Normal'),)):
    with (Path(root) / 'participants.tsv').open('w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['Group', 'participant_id'])
        for subject, group in entries:
            writer.writerow([group, subject])


def records(root, **options):
    return ADAuditoryDataset.set_records(ADAuditoryDataset.__new__(ADAuditoryDataset), str(root), **options)


def read_rows(record, **options):
    data = ADAuditoryDataset.read_record(record, **options)
    process = {key: options[key] for key in ['num_channel', 'trial_types', 'chunk_size', 'overlap', 'before_trial', 'offline_transform'] if key in options}
    return list(ADAuditoryDataset.process_record(record=record, **data, **process))


class TestADAuditoryDataset(unittest.TestCase):
    def test_native_header_fdt_values_event_bounds_and_diagnosis_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);participants(root)
            path, source, _ = write_record(root)
            record = records(root)[0]
            rows = read_rows(record, chunk_size=-1)
            self.assertEqual([row['eeg'].shape for row in rows], [(19, 500), (19, 250), (19, 500), (19, 125)])
            np.testing.assert_array_equal(rows[0]['eeg'], source[:, 250:750])
            np.testing.assert_array_equal(rows[-1]['eeg'], source[:, 1500:1625])
            self.assertEqual(rows[0]['info']['start_at'], 250)
            self.assertEqual(rows[-1]['info']['end_at'], 1625)
            self.assertEqual({row['info']['label'] for row in rows}, {0})
            self.assertEqual([row['info']['value'] for row in rows], [2, 1, 2, 1])
            self.assertEqual(rows[0]['info']['signal_unit'], 'uV')
            self.assertEqual(json.loads(rows[0]['info']['channel_names']), AD_AUDITORY_CHANNELS)
            self.assertEqual(record['set_path'], str(path))

    def test_discovery_keeps_every_paired_record_and_excludes_only_unknown_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            participants(root, [('sub-01','Normal'), ('sub-02','Mild AD'), ('sub-03','Moderate AD'), ('sub-04','MCI'), ('sub-05','-')])
            for subject in range(1, 6):
                write_record(root, f'sub-{subject:02d}', run='01')
            write_record(root, 'sub-01', run='02')
            result = records(root)
            self.assertEqual(len(result), 5)
            self.assertEqual([row['label'] for row in result], [0, 0, 1, 1, 2])
            self.assertEqual(len({row['record_id'] for row in result}), 5)
            all_rows = [row for record in result for row in read_rows(record, chunk_size=-1)]
            self.assertEqual(len({row['info']['trial_id'] for row in all_rows}), 20)
            self.assertEqual(len({row['key'] for row in all_rows}), 20)

    def test_event_selection_preserves_source_indices_and_whole_mode_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);participants(root);write_record(root)
            record = records(root)[0]
            rows = read_rows(record, trial_types=['Rest'], chunk_size=-1)
            self.assertEqual([row['info']['trial_index'] for row in rows], [1, 3])
            whole = read_rows(record, trial_types=None, chunk_size=-1)
            self.assertEqual(whole[0]['eeg'].shape, (19, 2500))
            self.assertEqual(whole[0]['info']['trial_type'], 'Whole')
            Path(record['events_path']).unlink()
            with self.assertRaises(FileNotFoundError):
                records(root)
            whole_record = records(root, trial_types=None)[0]
            self.assertEqual(read_rows(whole_record, trial_types=None, chunk_size=-1)[0]['eeg'].shape, (19, 2500))
            fake = ADAuditoryDataset.fake_record()
            fake['events'] = []
            with self.assertRaises(ValueError):
                list(ADAuditoryDataset.process_record(**fake, trial_types=['Stimulus']))

    def test_header_channel_permutation_is_reordered_to_the_documented_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);participants(root)
            _, source, _ = write_record(root, permutation=list(reversed(range(19))))
            row = read_rows(records(root)[0])[0]
            np.testing.assert_array_equal(row['eeg'], source[:, 250:500])

    def test_continuous_filters_reference_resampling_and_actual_output_times(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);participants(root);_, source, _ = write_record(root)
            options = dict(num_channel=3, sfreq=137, notch_freqs=50., l_freq=4., h_freq=38., ref_channels='average', chunk_size=-1)
            row = read_rows(records(root)[0], **options)[0]
            raw = mne.io.RawArray(source.astype(np.float64) * 1e-6,
                  mne.create_info(AD_AUDITORY_CHANNELS, 250, 'eeg'), verbose=False)
            raw.notch_filter(50., verbose=False).filter(4., 38., verbose=False)
            raw.set_eeg_reference('average', verbose=False)
            raw.resample(137, verbose=False)
            np.testing.assert_array_equal(row['eeg'], raw.get_data()[:3, 137:411] * 1e6)
            self.assertEqual(row['info']['source_start_seconds'], 1.)
            self.assertEqual(row['info']['source_end_seconds'], 3.)
            self.assertEqual(row['info']['sfreq'], 137)

    def test_invalid_headers_fdt_bytes_and_nonfinite_signals_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);participants(root)
            for key, value in [('nbchan',18), ('srate',128), ('trials',2), ('pnts',2501)]:
                path, _, _ = write_record(root)
                with h5py.File(path, 'a') as file:
                    file[key][0,0] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    read_rows(records(root)[0])
            path, _, _ = write_record(root)
            with path.with_suffix('.fdt').open('ab') as file:
                file.write(b'bad')
            with self.assertRaises(ValueError):
                read_rows(records(root)[0])
            path, source, _ = write_record(root)
            source[0,0] = np.nan;source.T.tofile(path.with_suffix('.fdt'))
            with self.assertRaises(ValueError):
                read_rows(records(root)[0])

    def test_invalid_events_are_not_clipped_or_replaced_with_whole_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);participants(root);path, _, events = write_record(root)
            cases = [dict(onset=-1), dict(onset=np.nan), dict(duration=0), dict(duration=100),
                     dict(trial_type='Other'), dict(sample=250), dict(value=1)]
            for changes in cases:
                values = copy.deepcopy(events);values[0].update(changes);write_events(path, values)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    read_rows(records(root)[0])
            values = copy.deepcopy(events);values[1]['onset']=2.;write_events(path,values)
            with self.assertRaises(ValueError):
                read_rows(records(root)[0])
            write_events(path, [])
            with self.assertRaises(ValueError):
                read_rows(records(root)[0])

    def test_participant_validation_prevents_silent_cohort_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_record(root)
            for entries in [[('sub-01','Normal'),('sub-01','MCI')], [('sub-01','Unknown')], [('invalid','Normal')]]:
                participants(root,entries)
                with self.subTest(entries=entries),self.assertRaises(ValueError):
                    records(root)
            participants(root,[('sub-02','Normal')])
            with self.assertRaises(FileNotFoundError):
                records(root)
            (root/'participants.tsv').write_text('participant_id\nsub-01\n')
            with self.assertRaises(ValueError):
                records(root)

    def test_overlapping_hooks_do_not_modify_sources_or_neighboring_windows(self):
        values = ADAuditoryDataset.fake_record()
        source = values['eeg_data'].copy()
        def before(eeg):
            eeg += 2
            return eeg
        def offline(eeg):
            eeg += 3
            return dict(eeg=eeg)
        rows = list(ADAuditoryDataset.process_record(**values, chunk_size=250, overlap=125,
                    before_trial=before, offline_transform=offline))
        np.testing.assert_array_equal(values['eeg_data'], source)
        np.testing.assert_array_equal(rows[1]['eeg'], source[:, 1375:1625] + 2 + 3)
        for hook in [lambda x:x[:,:-1], lambda x:np.full(x.shape,np.nan), lambda x:x.astype(complex)]:
            with self.assertRaises(ValueError):
                list(ADAuditoryDataset.process_record(**values, before_trial=hook))

    def test_cache_online_labels_and_parameter_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);participants(root);write_record(root)
            options=dict(root_path=directory,io_path=str(root/'cache'),io_mode='pickle',verbose=False,
                         online_transform=transforms.ToTensor(),label_transform=transforms.Select('label'))
            dataset=ADAuditoryDataset(**options)
            self.assertEqual(len(dataset),5)
            self.assertEqual(tuple(dataset[0][0].shape),(19,250))
            self.assertEqual(dataset[0][1],0)
            reused=ADAuditoryDataset(**options)
            np.testing.assert_array_equal(dataset[0][0],reused[0][0])
            with self.assertRaises(ValueError):
                ADAuditoryDataset(**options,trial_types=['Rest'])

    def test_constants_preserve_channel_order_and_all_grid_positions(self):
        from torcheeg.datasets.constants import (AD_AUDITORY_CHANNEL_LIST,
            AD_AUDITORY_CHANNEL_LOCATION_DICT,AD_AUDITORY_ADJACENCY_MATRIX)
        self.assertEqual(AD_AUDITORY_CHANNEL_LIST,AD_AUDITORY_CHANNELS)
        self.assertEqual(len({tuple(x) for x in AD_AUDITORY_CHANNEL_LOCATION_DICT.values()}),19)
        self.assertEqual(np.shape(AD_AUDITORY_ADJACENCY_MATRIX),(19,19))
        self.assertTrue(np.isfinite(AD_AUDITORY_ADJACENCY_MATRIX).all())

    def test_invalid_settings_fail_before_io_and_fake_data_does_not_change_rng(self):
        cases=[dict(trial_types=[]),dict(trial_types=['Rest','Rest']),dict(trial_types='Rest'),
               dict(sfreq=0),dict(sfreq=True),dict(num_channel=20),dict(overlap=-1),dict(overlap=250),
               dict(h_freq=125),dict(notch_freqs=np.nan),dict(l_freq=40,h_freq=4),dict(ref_channels='Cz')]
        for options in cases:
            with self.subTest(options=options),patch('torcheeg.datasets.module.clinical.ad_auditory.BaseDataset.__init__') as parent:
                with self.assertRaises(ValueError):
                    ADAuditoryDataset(root_path='unused',**options)
                parent.assert_not_called()
        state=np.random.get_state();fake=ADAuditoryDataset.fake_record();after=np.random.get_state()
        np.testing.assert_array_equal(state[1],after[1])
        self.assertEqual(fake['events'][-1]['onset']+fake['events'][-1]['duration'],350)
        self.assertEqual(next(ADAuditoryDataset.process_record(**fake))['eeg'].shape,(19,250))

    def test_import_does_not_change_warning_filters_or_mne_logging(self):
        import torcheeg.datasets.module.clinical.ad_auditory as module
        filters=list(warnings.filters);level=mne.utils.logger.level
        importlib.reload(module)
        self.assertEqual(warnings.filters,filters)
        self.assertEqual(mne.utils.logger.level,level)


if __name__ == '__main__':
    unittest.main()
