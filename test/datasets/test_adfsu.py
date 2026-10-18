import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import mne
import numpy as np

from torcheeg import transforms
from torcheeg.datasets import ADFSUDataset
from torcheeg.datasets.constants import ADFSU_ADJACENCY_MATRIX, ADFSU_CHANNEL_LOCATION_DICT
from torcheeg.datasets.module.clinical.adfsu import ADFSU_CHANNEL_LIST


def write_condition(root, group='Healthy', subject='Paciente1', state='Eyes_closed'):
    folder = Path(root) / group / state / subject
    folder.mkdir(parents=True, exist_ok=True)
    time = np.arange(1024) / 128.
    source = np.stack([(i + 1) * np.sin(2 * np.pi * (i + 2) * time) + .5 * i for i in range(19)])
    for channel, values in zip(ADFSU_CHANNEL_LIST, source):
        np.savetxt(folder / (channel + '.txt'), values, fmt='%.18e')
    return folder, source


def records(root, **options):
    return ADFSUDataset.set_records(ADFSUDataset.__new__(ADFSUDataset), str(root), **options)


def rows(record, root, **options):
    data = ADFSUDataset.read_record(record, root_path=str(root), **options)
    process = {k: v for k, v in options.items() if k in ['chunk_size','overlap','num_channel','before_trial','offline_transform']}
    return list(ADFSUDataset.process_record(record=record, **data, **process))


class TestADFSUDataset(unittest.TestCase):
    def test_native_values_labels_channels_units_and_complete_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, source = write_condition(root)
            write_condition(root, state='Eyes_open')
            write_condition(root, 'AD', 'Paciente12')
            found = records(root)
            self.assertEqual([r['subject_id'] for r in found], ['Healthy_Paciente1', 'AD_Paciente12'])
            values = rows(found[0], root)
            self.assertEqual(len(values), 16)
            for i, item in enumerate(values):
                start = (i % 8) * 128
                np.testing.assert_array_equal(item['eeg'], source[:, start:start+128])
                self.assertEqual(item['info']['label'], 0)
                self.assertEqual(item['info']['signal_unit'], 'source')
                self.assertEqual(json.loads(item['info']['channel_names']), ADFSU_CHANNEL_LIST)
                self.assertEqual(item['info']['source_start_seconds'], start / 128)
            self.assertEqual(len({v['key'] for v in values}), 16)
            self.assertEqual(len({v['info']['trial_id'] for v in values}), 2)
            self.assertEqual({v['info']['label'] for v in rows(found[1], root)}, {1})
            np.testing.assert_array_equal(rows(found[0], root, chunk_size=-1)[0]['eeg'], source)

    def test_discovery_natural_sort_same_folder_ids_are_conservative_and_state_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for group, subject in [('AD','Paciente10'),('AD','Paciente2'),('Healthy','Paciente2')]:
                for state in ['Eyes_closed','Eyes_open']:
                    write_condition(root, group, subject, state)
            found = records(root)
            self.assertEqual([r['subject_id'] for r in found], ['Healthy_Paciente2','AD_Paciente2','AD_Paciente10'])
            both = rows(found[1], root, chunk_size=-1)
            rest = rows(records(root, states=['Eyes_open'])[1], root, states=['Eyes_open'], chunk_size=-1)
            self.assertEqual(rest[0]['info']['trial_id'], both[1]['info']['trial_id'])
            self.assertEqual(rest[0]['key'], both[1]['key'])
            self.assertEqual(both[0]['info']['subject_id'], both[1]['info']['subject_id'])
            self.assertNotEqual(both[0]['info']['record_id'], both[1]['info']['record_id'])

    def test_empty_condition_warns_but_partial_channels_raise_even_when_unselected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, _ = write_condition(root)
            empty = root / 'Healthy' / 'Eyes_open' / 'Paciente1';empty.mkdir(parents=True)
            with self.assertWarnsRegex(UserWarning, 'Empty ADFSU condition'):
                found = records(root)
            self.assertEqual(found[0]['states'], ('Eyes_closed',))
            np.savetxt(empty / 'Fp1.txt', np.zeros(1024))
            with self.assertRaises(FileNotFoundError):
                records(root)
            (empty / 'Fp1.txt').unlink();empty.rmdir()
            (folder / 'O2.txt').unlink()
            with self.assertRaises(FileNotFoundError):
                records(root)
            record = dict(group='Healthy',subject='Paciente1',subject_id='Healthy_Paciente1',states=('Eyes_closed',))
            with self.assertRaises(FileNotFoundError):
                ADFSUDataset.read_record(record,root_path=root,num_channel=1)

    def test_invalid_settings_do_not_create_cache_or_enter_nonpositive_window_step(self):
        cases = [dict(overlap=128),dict(overlap=-1),dict(chunk_size=True),dict(chunk_size=2.5),
                 dict(sfreq=0),dict(sfreq=False),dict(num_channel=20),dict(num_channel=0),
                 dict(states=[]),dict(states=['Eyes_open','Eyes_open']),dict(states='Eyes_open'),
                 dict(states=['other']),dict(l_freq=64),dict(h_freq=float('nan')),
                 dict(l_freq=20,h_freq=10),dict(notch_freqs=-1),dict(ref_channels='Cz')]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'cache'
            for options in cases:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    ADFSUDataset(io_path=str(cache),**options)
                self.assertFalse(cache.exists())
        fake=ADFSUDataset.fake_record()
        with self.assertRaises(ValueError):
            list(ADFSUDataset.process_record(**fake, chunk_size=-1,overlap=1024))

    def test_strict_source_geometry_finiteness_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, source=write_condition(directory)
            record=records(directory)[0]
            for values in [np.zeros(1023),np.zeros((512,2)),np.full(1024,np.nan),np.full(1024,np.inf)]:
                with self.subTest(shape=values.shape):
                    np.savetxt(folder/'Fp1.txt',values)
                    with self.assertRaises(ValueError):
                        ADFSUDataset.read_record(record,root_path=directory)
            np.savetxt(folder/'Fp1.txt',source[0])
            bad=dict(record,subject='../Paciente1')
            with self.assertRaises(ValueError):
                ADFSUDataset.read_record(bad,root_path=directory)
            bad=dict(record,subject_id='AD_Paciente1')
            with self.assertRaises(ValueError):
                ADFSUDataset.read_record(bad,root_path=directory)
            fake=ADFSUDataset.fake_record()
            for field,value in [('subject_id','other'),('states',['Eyes_closed']),('samples',[np.zeros((19,0)),np.zeros((19,0))])]:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    list(ADFSUDataset.process_record(**dict(fake,**{field:value})))

    def test_preprocess_all_19_before_subset_with_independent_mne_and_output_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            _, source=write_condition(directory)
            record=records(directory)[0]
            options=dict(sfreq=100,notch_freqs=50.,l_freq=2.,h_freq=30.,ref_channels='average',num_channel=3)
            raw=mne.io.RawArray(source.copy(),mne.create_info(ADFSU_CHANNEL_LIST,128,'eeg'),verbose=False)
            raw.notch_filter(50.,verbose=False);raw.filter(2.,30.,verbose=False)
            raw.set_eeg_reference('average',verbose=False);raw.resample(100,verbose=False)
            result=rows(record,directory,chunk_size=100,**options)
            self.assertEqual(len(result),8)
            for index,item in enumerate(result):
                np.testing.assert_array_equal(item['eeg'],raw.get_data()[:3,index*100:(index+1)*100])
                self.assertEqual(item['info']['sfreq'],100)
                self.assertEqual(item['info']['source_sfreq'],128)
                self.assertEqual(item['info']['source_start_seconds'],index)
            mean=rows(record,directory,num_channel=1,ref_channels='average',chunk_size=-1)[0]['eeg']
            np.testing.assert_allclose(mean,source[:1]-source.mean(axis=0),atol=1e-12)

    def test_hooks_copy_overlap_windows_and_preserve_metadata_geometry(self):
        fake=ADFSUDataset.fake_record()
        before=[x.copy() for x in fake['samples']]
        def change(eeg):
            eeg[:]=eeg+10
            return {'eeg':eeg}
        result=list(ADFSUDataset.process_record(**fake,chunk_size=128,overlap=64,offline_transform=change))
        self.assertEqual(len(result),30)
        np.testing.assert_array_equal(result[1]['eeg'],before[0][:,64:192]+10)
        result[0]['eeg'][:]=0
        np.testing.assert_array_equal(result[1]['eeg'],before[0][:,64:192]+10)
        for actual,expected in zip(fake['samples'],before):np.testing.assert_array_equal(actual,expected)
        def center(values):
            values-=values.mean(axis=-1,keepdims=True)
            return values
        list(ADFSUDataset.process_record(**fake,before_trial=center))
        for actual,expected in zip(fake['samples'],before):np.testing.assert_array_equal(actual,expected)
        for hook in [lambda x:x[:,:-1],lambda x:np.full_like(x,np.nan)]:
            with self.assertRaises(ValueError):list(ADFSUDataset.process_record(**fake,before_trial=hook))

    def test_dataset_cache_transforms_and_independent_views(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'source';write_condition(root)
            cache=Path(directory)/'cache'
            dataset=ADFSUDataset(root_path=str(root),io_path=str(cache),io_mode='pickle',verbose=False,
                                online_transform=transforms.ToTensor(),label_transform=transforms.Select('label'))
            self.assertEqual(len(dataset),8)
            x,y=dataset[0];self.assertEqual(tuple(x.shape),(19,128));self.assertEqual(y,0)
            reopened=ADFSUDataset(root_path=str(root),io_path=str(cache),io_mode='pickle',verbose=False)
            np.testing.assert_array_equal(reopened[0][0],dataset.read_eeg(str(dataset.info.iloc[0]['_record_id']),str(dataset.info.iloc[0]['clip_id'])))
            view=copy.copy(dataset);view.info=view.info.iloc[:2].copy();self.assertEqual(len(dataset),8)
            self.assertIn('states',repr(dataset))
            memory=ADFSUDataset(root_path=str(root),io_path=str(Path(directory)/'unused'),io_mode='memory',verbose=False)
            self.assertEqual(len(memory),8);self.assertFalse((Path(directory)/'unused').exists())

    def test_global_rng_warning_policy_and_template_graph_unchanged(self):
        state=np.random.get_state()
        import torcheeg.datasets.module.clinical.adfsu as module
        filters=list(warnings.filters);importlib.reload(module)
        self.assertEqual(warnings.filters,filters)
        first=ADFSUDataset.fake_record();second=ADFSUDataset.fake_record()
        np.testing.assert_array_equal(first['samples'][0],second['samples'][0])
        after=np.random.get_state()
        for a,b in zip(state,after):np.testing.assert_equal(a,b)
        self.assertEqual(np.asarray(ADFSU_ADJACENCY_MATRIX).shape,(19,19))
        self.assertEqual(list(ADFSU_CHANNEL_LOCATION_DICT),ADFSU_CHANNEL_LIST)
        self.assertEqual(len(set(tuple(value) for value in ADFSU_CHANNEL_LOCATION_DICT.values())),19)

    def test_empty_selection_and_bad_subject_name_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):records(directory)
            write_condition(directory,subject='Paciente0')
            with self.assertRaises(ValueError):records(directory)
        with self.assertRaises(FileNotFoundError):records('/not/an/adfsu/root')


if __name__=='__main__':
    unittest.main()
