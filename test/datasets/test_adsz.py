import importlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import mne
import numpy as np

from torcheeg import transforms
from torcheeg.datasets import ADSZDataset
from torcheeg.datasets.constants import (ADSZ_CHANNEL_LIST, ADSZ_CHANNEL_LOCATION_DICT,
    ADSZ_ADJACENCY_MATRIX, ADSZ_SCHIZOPHRENIA_CHANNEL_LIST,
    ADSZ_SCHIZOPHRENIA_CHANNEL_LOCATION_DICT, ADSZ_SCHIZOPHRENIA_ADJACENCY_MATRIX)


AD_CHANNELS = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']
SZ_CHANNELS = ['F7','F3','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']


def write_source(root, subset='alzheimer', rows=None):
    folder = Path(root) / 'dataset' / subset
    names = [('Healthy','eeg20c1.out'),('Healthy','eeg20o1.out'),('AD','ec0101.out'),('AD','eo0101.out')] if subset == 'alzheimer' else [('Healthy','S10W1.eea'),('SCH','387-02w1.eea'),('SCH','387-03w1.eea')]
    signals = {}
    for index, (group, name) in enumerate(names if rows is None else rows):
        path = folder / group / name
        path.parent.mkdir(parents=True, exist_ok=True)
        channels, samples = (19, 1536) if subset == 'alzheimer' else (16, 7680)
        time = np.arange(samples) / 128
        values = np.stack([(channel + 1) * np.sin(2 * np.pi * (channel + 2) * time) + index + channel for channel in range(channels)])
        np.savetxt(path, values.T if subset == 'alzheimer' else values.reshape(-1, 1), fmt='%.18e')
        signals[name] = values
    return folder, signals


def rows(record, **options):
    data = ADSZDataset.read_record(record, **options)
    return list(ADSZDataset.process_record(**dict(data, **{k: v for k, v in options.items() if k in ['chunk_size','overlap']})))


class TestADSZDataset(unittest.TestCase):
    def test_native_two_formats_channel_order_values_units_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            for subset in ['alzheimer','schizophrenia']:
                _, source = write_source(directory, subset)
                records = ADSZDataset.set_records(directory, subset=subset)
                for record in records:
                    result = rows(record, chunk_size=128)
                    expected = source[record['file_name']] * (1 if subset == 'alzheimer' else 1e-6)
                    self.assertEqual(len(result), 12 if subset == 'alzheimer' else 60)
                    for index, item in enumerate(result):
                        np.testing.assert_array_equal(item['eeg'], expected[:, index*128:(index+1)*128])
                        info = item['info']
                        self.assertEqual(info['label'], 0 if record['class_name'] == 'HC' else 1)
                        self.assertEqual(info['source_start_seconds'], index)
                        self.assertEqual(info['source_end_seconds'], index + 1)
                        self.assertEqual(info['signal_unit'], 'source' if subset == 'alzheimer' else 'V')
                        self.assertEqual(json.loads(info['channel_names']), AD_CHANNELS if subset == 'alzheimer' else SZ_CHANNELS)
                    self.assertEqual(len({item['key'] for item in result}), len(result))

    def test_stable_conservative_ad_groups_and_provider_sz_file_subjects(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, _ = write_source(directory)
            before = ADSZDataset.set_records(directory)
            self.assertEqual(len({r['subject_id'] for r in before}), 2)
            self.assertEqual(len({r['record_id'] for r in before}), 4)
            self.assertTrue(all(r['identity_basis'] == 'conservative_filename_group_not_clinically_verified' for r in before))
            np.savetxt(folder/'AD/ec0001.out', np.zeros((128,19)))
            after = ADSZDataset.set_records(directory)
            self.assertEqual({r['file_name']:r['subject_id'] for r in before}, {r['file_name']:r['subject_id'] for r in after if r['file_name'] != 'ec0001.out'})
            write_source(directory,'schizophrenia')
            sz = ADSZDataset.set_records(directory,subset='schizophrenia')
            self.assertEqual(len({r['subject_id'] for r in sz}),3)
            self.assertTrue(all(r['identity_basis']=='provider_one_file_per_subject' for r in sz))

    def test_full_dataset_or_selected_roots_and_ambiguous_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder,_=write_source(root)
            expected=ADSZDataset.set_records(root)
            self.assertEqual(ADSZDataset.set_records(root/'dataset'),expected)
            self.assertEqual(ADSZDataset.set_records(folder),expected)
            write_source(root/'other')
            for group in ['AD','Healthy']:
                (root/group).mkdir()
            with self.assertRaisesRegex(ValueError,'Multiple ADSZ'):ADSZDataset.set_records(root)

    def test_explicit_unit_override_and_variable_lengths_and_channel_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            _,source=write_source(directory)
            record=ADSZDataset.set_records(directory)[0]
            for unit,scale in [('source',1),('uV',1e-6),('V',1)]:
                item=rows(record,source_unit=unit,num_channel=3,chunk_size=-1)[0]
                np.testing.assert_array_equal(item['eeg'],source[record['file_name']][:3]*scale)
                self.assertEqual(item['info']['units_basis'],'caller_supplied_unit')
                self.assertEqual(item['info']['end_at'],1536)
                self.assertEqual(len(json.loads(item['info']['channel_names'])),3)

    def test_filters_reference_then_resampling_use_all_channels_before_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            for subset in ['alzheimer','schizophrenia']:
                _,source=write_source(directory,subset)
                record=ADSZDataset.set_records(directory,subset=subset)[0]
                values=source[record['file_name']]*(1 if subset=='alzheimer' else 1e-6)
                channels=AD_CHANNELS if subset=='alzheimer' else SZ_CHANNELS
                raw=mne.io.RawArray(values.copy(),mne.create_info(channels,128,'eeg'),verbose=False)
                raw.notch_filter(50.,verbose=False);raw.filter(1.,40.,verbose=False)
                raw.set_eeg_reference('average',verbose=False);raw.resample(137,verbose=False)
                actual=rows(record,num_channel=3,sfreq=137,notch_freqs=50.,l_freq=1.,h_freq=40.,ref_channels='average',chunk_size=-1)[0]['eeg']
                np.testing.assert_array_equal(actual,raw.get_data()[:3])

    def test_invalid_settings_fail_before_cache_creation(self):
        bad=[dict(subset='mixed'),dict(subset=[]),dict(num_channel=0),dict(num_channel=20),dict(subset='schizophrenia',num_channel=19),dict(sfreq=0),dict(sfreq=True),dict(chunk_size=1.5),dict(overlap=-1),dict(overlap=128),dict(l_freq=64),dict(h_freq=float('nan')),dict(l_freq=20,h_freq=10),dict(ref_channels='Cz'),dict(source_unit='mV')]
        with tempfile.TemporaryDirectory() as directory:
            for options in bad:
                with self.subTest(options=options):
                    cache=Path(directory)/'unused'
                    with self.assertRaises(ValueError):ADSZDataset(root_path='absent',io_path=str(cache),**options)
                    self.assertFalse(cache.exists())

    def test_malformed_matrices_columns_nonfinite_and_metadata_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            for subset in ['alzheimer','schizophrenia']:
                write_source(directory,subset)
                record=ADSZDataset.set_records(directory,subset=subset)[0]
                path=Path(record['file_path'])
                original=path.read_bytes()
                malformed=[np.zeros((3,18)),np.zeros((16,7680)),np.full((128,19),np.nan)]
                if subset=='schizophrenia':malformed.append(np.full((122880,1),np.nan))
                for values in malformed:
                    np.savetxt(path,values)
                    with self.assertRaises(ValueError):ADSZDataset.read_record(record)
                path.write_bytes(original)
                with self.assertRaises(ValueError):ADSZDataset.read_record(dict(record,label=1-record['label']))
                with self.assertRaises(ValueError):ADSZDataset.read_record(record,subset='schizophrenia' if subset=='alzheimer' else 'alzheimer')
                path.unlink()
                with self.assertRaises(FileNotFoundError):ADSZDataset.read_record(record)

    def test_missing_empty_unknown_and_duplicate_sources_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(FileNotFoundError):ADSZDataset.set_records(root)
            (root/'AD').mkdir();(root/'Healthy').mkdir()
            with self.assertRaises(ValueError):ADSZDataset.set_records(root)
            np.savetxt(root/'AD/wrong.out',np.zeros((128,19)))
            np.savetxt(root/'Healthy/eeg20c1.out',np.zeros((128,19)))
            with self.assertRaises(ValueError):ADSZDataset.set_records(root)
            (root/'AD/wrong.out').rename(root/'AD/ec0101.out')
            write_source(root/'sz','schizophrenia')
            p=root/'sz/dataset/schizophrenia/Healthy'
            np.savetxt(p/'s10w1.eea',np.zeros((122880,1)))
            original_glob=Path.glob
            def duplicate_paths(folder,pattern):
                found=list(original_glob(folder,pattern))
                return iter(found+[folder/'S10W1.eea',folder/'s10w1.eea']) if folder.resolve()==p.resolve() else iter(found)
            with patch.object(Path,'glob',autospec=True,side_effect=duplicate_paths):
                with self.assertRaisesRegex(ValueError,'Duplicate'):ADSZDataset.set_records(root/'sz',subset='schizophrenia')

    def test_overlapping_windows_and_hooks_are_independent_and_geometry_checked(self):
        source=np.arange(19*300,dtype=float).reshape(19,300)
        original=source.copy();fake=ADSZDataset.fake_record();fake['signal']=source
        def change(eeg):
            eeg[:]=-99
            return dict(eeg=eeg)
        result=list(ADSZDataset.process_record(**fake,chunk_size=128,overlap=64,offline_transform=change))
        self.assertEqual(len(result),3)
        np.testing.assert_array_equal(source,original)
        result[0]['eeg'][:]=0
        self.assertTrue((result[1]['eeg']==-99).all())
        for hook in [lambda x:x[:,:-1],lambda x:np.full_like(x,np.nan)]:
            with self.assertRaises(ValueError):list(ADSZDataset.process_record(**fake,before_trial=hook))
        with self.assertRaises(ValueError):list(ADSZDataset.process_record(**fake,chunk_size=-1,overlap=300))
        self.assertEqual(len(list(ADSZDataset.process_record(**fake,chunk_size=301))),0)

    def test_actual_pickle_and_memory_io_transforms_and_namespace_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for subset in ['alzheimer','schizophrenia']:
                _,source=write_source(root,subset)
                for mode in ['pickle','memory']:
                    cache=root/(subset+'-'+mode)
                    dataset=ADSZDataset(root_path=str(root),subset=subset,io_path=str(cache),io_mode=mode,verbose=False,
                        online_transform=transforms.Compose([transforms.ToTensor(),transforms.To2d()]),label_transform=transforms.Select('label'))
                    eeg,label=dataset[0]
                    self.assertEqual(tuple(eeg.shape),(1,19 if subset=='alzheimer' else 16,128))
                    self.assertEqual(label,0)
                    expected=source[dataset.info.iloc[0].file_name][:,:128]*(1 if subset=='alzheimer' else 1e-6)
                    np.testing.assert_allclose(eeg.numpy()[0],expected,rtol=1e-6,atol=1e-12)
                    self.assertEqual(dataset.info.record_id.nunique(),4 if subset=='alzheimer' else 3)
                    if mode=='memory':self.assertFalse(cache.exists())
                    else:
                        reused=ADSZDataset(root_path=str(root),subset=subset,io_path=str(cache),io_mode=mode,verbose=False,
                            online_transform=transforms.Compose([transforms.ToTensor(),transforms.To2d()]),label_transform=transforms.Select('label'))
                        self.assertEqual(len(reused),len(dataset))
                        with self.assertRaisesRegex(ValueError,'cache configuration'):
                            ADSZDataset(root_path='unavailable',subset=subset,io_path=str(cache),io_mode=mode,verbose=False)

    def test_constants_preserve_source_orders_locations_and_induced_graph(self):
        self.assertEqual(ADSZ_CHANNEL_LIST,AD_CHANNELS)
        self.assertEqual(ADSZ_SCHIZOPHRENIA_CHANNEL_LIST,SZ_CHANNELS)
        indices=[AD_CHANNELS.index(name) for name in SZ_CHANNELS]
        np.testing.assert_array_equal(ADSZ_SCHIZOPHRENIA_ADJACENCY_MATRIX,np.asarray(ADSZ_ADJACENCY_MATRIX)[np.ix_(indices,indices)])
        for name in SZ_CHANNELS:
            self.assertEqual(ADSZ_SCHIZOPHRENIA_CHANNEL_LOCATION_DICT[name],ADSZ_CHANNEL_LOCATION_DICT[name])

    def test_fake_and_module_reload_preserve_random_and_warning_state(self):
        state=np.random.get_state();filters=list(warnings.filters)
        for subset in ['alzheimer','schizophrenia']:
            fake=ADSZDataset.fake_record(subset=subset)
            item=next(ADSZDataset.process_record(**fake))
            self.assertEqual(item['eeg'].shape,(19 if subset=='alzheimer' else 16,128))
        now=np.random.get_state()
        self.assertEqual(state[0],now[0]);np.testing.assert_array_equal(state[1],now[1]);self.assertEqual(state[2:],now[2:])
        importlib.reload(importlib.import_module('torcheeg.datasets.module.clinical.adsz'))
        self.assertEqual(filters,warnings.filters)


if __name__=='__main__':unittest.main()
