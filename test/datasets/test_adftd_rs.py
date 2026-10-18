import copy
import csv
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import mne
import numpy as np
from scipy.io import loadmat, savemat

from torcheeg import transforms
from torcheeg.datasets import ADFTDRSDataset
from torcheeg.datasets.constants import ADFTD_RS_CHANNEL_LIST, ADFTD_RS_CHANNEL_LOCATION_DICT, ADFTD_RS_ADJACENCY_MATRIX
from torcheeg.datasets.module.clinical._adftd import CHANNELS


NATIVE = dict(sfreq=500, notch_freqs=None, l_freq=None, h_freq=None)


def participants(root, rows=(('sub-001','C'),), bom=False):
    path=Path(root)/'participants.tsv';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig' if bom else 'utf-8',newline='') as file:
        writer=csv.writer(file,delimiter='\t');writer.writerow(['participant_id','Group','MMSE  ']);writer.writerows((subject,group,30) for subject,group in rows)


def write_record(root, subject='sub-001', run=None, permutation=None, preprocessed=False, boundary=None, samples=6000, nested=False, external=False):
    folder=Path(root)/('derivatives' if preprocessed else '')/subject/'eeg';folder.mkdir(parents=True,exist_ok=True)
    identity=subject+'_task-eyesclosed'+(f'_run-{run}' if run else '')
    path=folder/(identity+'_eeg.set')
    time=np.arange(samples)/500.
    values=np.stack([(index+1)*np.sin(2*np.pi*(index+2)*time)+.2*index for index in range(19)]).astype(np.float32)
    order=list(range(19)) if permutation is None else permutation
    data=values[order]
    if external:
        data.T.astype('<f4').tofile(path.with_suffix('.fdt'));data=path.with_suffix('.fdt').name
    channels=np.array([dict(labels=CHANNELS[index].capitalize()) for index in order],dtype=object)
    events=[] if boundary is None else [dict(type='boundary',latency=boundary,duration=500.)]
    payload=dict(nbchan=19,srate=500.,pnts=samples,trials=1,xmin=0.,xmax=(samples-1)/500.,data=data,chanlocs=channels,event=np.array(events,dtype=object))
    savemat(path,{'EEG':payload} if nested else payload)
    rawfolder=Path(root)/subject/'eeg';rawfolder.mkdir(parents=True,exist_ok=True)
    cpath=rawfolder/(identity+'_channels.tsv')
    with cpath.open('w',newline='') as file:
        writer=csv.writer(file,delimiter='\t');writer.writerow(['name','type','units']);writer.writerows((CHANNELS[index],'EEG','microV') for index in order)
    (rawfolder/(identity+'_eeg.json')).write_text(json.dumps(dict(TaskName='eyesclosed',SamplingFrequency=500,RecordingDuration=samples/500.)))
    return path,values


def records(root, **options):
    return ADFTDRSDataset.set_records(root_path=str(root),**options)


def rows(record, options=None, **process):
    options=dict(NATIVE) if options is None else options
    data=ADFTDRSDataset.read_record(record,**options)
    if 'num_channel' in options:process['num_channel']=options['num_channel']
    return list(ADFTDRSDataset.process_record(record=record,**data,**process))


class TestADFTDRSDataset(unittest.TestCase):
    def test_actual_mat_values_units_channel_reorder_nested_and_external_data(self):
        for nested,external in [(False,False),(True,False),(False,True)]:
            with self.subTest(nested=nested,external=external),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);participants(root)
                _,source=write_record(root,permutation=list(reversed(range(19))),nested=nested,external=external)
                result=rows(records(root)[0],chunk_size=500)
                self.assertEqual(len(result),12)
                for i,item in enumerate(result):
                    np.testing.assert_array_equal(item['eeg'],source[:,i*500:(i+1)*500].astype(np.float64)*1e-6)
                    info=item['info'];self.assertEqual(info['label'],0);self.assertEqual(info['subject_id'],1)
                    self.assertEqual(info['participant_id'],'sub-001');self.assertEqual(info['signal_unit'],'V')
                    self.assertEqual(json.loads(info['channel_names']),CHANNELS)
                    self.assertEqual(info['source_start_seconds'],i);self.assertEqual(info['source_end_seconds'],i+1)
                    self.assertEqual(info['data_source'],'raw');self.assertEqual(info['source_clock'],'stored_recording')
                self.assertEqual(len({item['key'] for item in result}),12)

    def test_all_diagnoses_multiple_runs_and_bom_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);participants(root,[('sub-001','C'),('sub-002','A'),('sub-003','F')],bom=True)
            for subject in ['sub-001','sub-002','sub-003']:write_record(root,subject)
            write_record(root,'sub-001',run='02')
            found=records(root);self.assertEqual(len(found),4)
            result=[item for record in found for item in rows(record,chunk_size=-1)]
            self.assertEqual([item['info']['label'] for item in result],[0,0,1,2])
            self.assertEqual(len({item['info']['record_id'] for item in result}),4)
            self.assertEqual(len({item['key'] for item in result}),4)

    def test_known_participant_missing_data_or_sidecars_fails_without_silent_drop(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);participants(root,[('sub-001','C'),('sub-002','A')]);write_record(root)
            with self.assertRaises(FileNotFoundError):records(root)
            path,_=write_record(root,'sub-002');path.with_suffix('.json').unlink()
            with self.assertRaises(FileNotFoundError):records(root)

    def test_diagnosis_and_subject_ids_are_strict(self):
        cases=[[('sub-001','unknown')],[('sub-001','C'),('sub-001','A')],[('sub-1','C'),('sub-001','A')],[('../sub-001','C')],[('sub-000','C')],[]]
        with tempfile.TemporaryDirectory() as directory:
            for case in cases:
                with self.subTest(case=case):
                    participants(directory,case)
                    with self.assertRaises(ValueError):records(directory)

    def test_default_pipeline_matches_independent_full_19_channel_mne(self):
        with tempfile.TemporaryDirectory() as directory:
            participants(directory);_,source=write_record(directory)
            raw=mne.io.RawArray(source.astype(np.float64)*1e-6,mne.create_info(CHANNELS,500,'eeg'),verbose=False)
            raw.notch_filter(50.,verbose=False);raw.filter(.5,45.,verbose=False);raw.resample(200,verbose=False)
            result=rows(records(directory)[0],options={},chunk_size=400)
            self.assertEqual(len(result),6)
            for index,item in enumerate(result):np.testing.assert_array_equal(item['eeg'],raw.get_data()[:,index*400:(index+1)*400])
            self.assertEqual(result[-1]['info']['end_at'],2400)
            self.assertEqual(result[-1]['info']['source_end_seconds'],12.)
            subset=rows(records(directory)[0],options=dict(NATIVE,num_channel=3,ref_channels='average'),chunk_size=-1)[0]['eeg']
            np.testing.assert_allclose(subset,(source.astype(np.float64)-source.astype(np.float64).mean(0))[:3]*1e-6,atol=1e-19)

    def test_preprocessed_source_and_half_sample_boundary_do_not_mix_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);participants(root)
            _,raw_values=write_record(root)
            path,clean=write_record(root,preprocessed=True,boundary=2000.5,samples=5000)
            # The raw sidecar describes the original 12-second recording, not ASR's compressed clock.
            (root/'sub-001/eeg/sub-001_task-eyesclosed_eeg.json').write_text(json.dumps(dict(TaskName='eyesclosed',SamplingFrequency=500,RecordingDuration=12.)))
            record=records(root,preprocessed=True)[0]
            output=rows(record,options=dict(NATIVE,preprocessed=True),chunk_size=1500)
            self.assertEqual(len(output),3)
            for actual,(first,last) in zip(output,[(0,1500),(2000,3500),(3500,5000)]):
                np.testing.assert_array_equal(actual['eeg'],clean[:,first:last].astype(np.float64)*1e-6)
                self.assertEqual(actual['info']['source_start_seconds'],first/500)
            self.assertEqual([item['info']['segment_index'] for item in output],[0,1,1])
            self.assertEqual({item['info']['data_source'] for item in output},{'preprocessed'})
            native=rows(records(root)[0],chunk_size=-1)[0]
            self.assertNotEqual(native['info']['record_id'],output[0]['info']['record_id'])
            self.assertEqual(native['info']['subject_id'],output[0]['info']['subject_id'])
            with self.assertRaises(ValueError):rows(record)

    def test_boundary_filtering_and_resampling_are_per_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            participants(directory);_,source=write_record(directory,boundary=3000.5)
            record=records(directory)[0]
            result=rows(record,options=dict(sfreq=137,notch_freqs=None,l_freq=4.,h_freq=40.),chunk_size=-1)
            self.assertEqual(len(result),2)
            for index,item in enumerate(result):
                raw=mne.io.RawArray(source[:,index*3000:(index+1)*3000].astype(np.float64)*1e-6,mne.create_info(CHANNELS,500,'eeg'),verbose=False)
                raw.filter(4.,40.,verbose=False);raw.resample(137,verbose=False)
                np.testing.assert_array_equal(item['eeg'],raw.get_data())
                self.assertEqual(item['info']['source_start_seconds'],index*6.)
                self.assertEqual(item['info']['source_end_seconds'],(index+1)*6.)

    def test_resampling_half_sample_ties_and_minimum_one_sample_follow_mne(self):
        for samples,rate,count in [(3250,137,891),(10,1,1)]:
            with self.subTest(samples=samples,rate=rate),tempfile.TemporaryDirectory() as directory:
                participants(directory);_,source=write_record(directory,samples=samples)
                record=records(directory)[0]
                result=rows(record,options=dict(NATIVE,sfreq=rate),chunk_size=-1)
                raw=mne.io.RawArray(source.astype(np.float64)*1e-6,mne.create_info(CHANNELS,500,'eeg'),verbose=False)
                raw.resample(rate,verbose=False)
                self.assertEqual(len(result),1)
                self.assertEqual(result[0]['eeg'].shape,(19,count))
                np.testing.assert_array_equal(result[0]['eeg'],raw.get_data())
                self.assertEqual(result[0]['info']['source_end_seconds'],samples/500.)

    def test_invalid_settings_fail_before_cache_creation_and_before_window_loop(self):
        cases=[dict(overlap=400),dict(overlap=-1),dict(chunk_size=True),dict(num_channel=0),dict(num_channel=20),dict(sfreq=0),dict(sfreq=200.5),dict(notch_freqs=250),dict(l_freq=float('nan')),dict(l_freq=50,h_freq=20),dict(ref_channels='Cz'),dict(preprocessed='yes')]
        with tempfile.TemporaryDirectory() as directory:
            for options in cases:
                with self.subTest(options=options),self.assertRaises(ValueError):ADFTDRSDataset(io_path=str(Path(directory)/'cache'),**options)
                self.assertFalse((Path(directory)/'cache').exists())
        with self.assertRaises(ValueError):list(ADFTDRSDataset.process_record(**ADFTDRSDataset.fake_record(),chunk_size=-1,overlap=1600))

    def test_corrupt_header_geometry_and_boundary_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            participants(directory);path,_=write_record(directory);record=records(directory)[0]
            original={key:value for key,value in loadmat(path,simplify_cells=True).items() if not key.startswith('__')}
            cases=[dict(srate=250),dict(trials=2),dict(nbchan=18),dict(pnts=1),dict(xmin=1),dict(xmax=99),dict(data=np.full((19,6000),np.nan)),dict(event=np.array([dict(type='boundary',latency=-1)],dtype=object))]
            for changes in cases:
                with self.subTest(keys=list(changes)):
                    savemat(path,dict(original,**changes))
                    with self.assertRaises(ValueError):ADFTDRSDataset.read_record(record,**NATIVE)

    def test_sidecar_duration_task_sampling_and_channel_metadata_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            participants(directory);path,_=write_record(directory);record=records(directory)[0]
            original=json.loads(path.with_suffix('.json').read_text())
            for changes in [dict(SamplingFrequency=200),dict(TaskName='photomark'),dict(EEGChannelCount=18),dict(RecordingType='epoched'),dict(RecordingDuration=10),dict(RecordingDuration=float('nan'))]:
                with self.subTest(changes=changes):
                    path.with_suffix('.json').write_text(json.dumps(dict(original,**changes)))
                    with self.assertRaises(ValueError):ADFTDRSDataset.read_record(record,**NATIVE)
            path.with_suffix('.json').write_text(json.dumps(original))
            Path(record['channels_path']).write_text('name\ttype\tunits\nFP1\tEEG\tV\n')
            with self.assertRaises(ValueError):ADFTDRSDataset.read_record(record,**NATIVE)

    def test_hooks_and_overlapping_windows_do_not_mutate_inputs_or_other_windows(self):
        fake=ADFTDRSDataset.fake_record();before=fake['segments'][0]['eeg'].copy()
        def modify(eeg):eeg+=1;return {'eeg':eeg}
        result=list(ADFTDRSDataset.process_record(**fake,chunk_size=400,overlap=200,offline_transform=modify))
        np.testing.assert_array_equal(result[1]['eeg'],before[:,200:600]+1)
        result[0]['eeg'][:]=0
        np.testing.assert_array_equal(result[1]['eeg'],before[:,200:600]+1)
        np.testing.assert_array_equal(fake['segments'][0]['eeg'],before)
        for hook in [lambda x:x[:,:-1],lambda x:np.full_like(x,np.nan)]:
            with self.assertRaises(ValueError):list(ADFTDRSDataset.process_record(**fake,before_trial=hook))

    def test_cache_online_label_views_and_memory_no_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'source';participants(root);_,source=write_record(root)
            cache=Path(directory)/'cache'
            dataset=ADFTDRSDataset(root_path=str(root),io_path=str(cache),io_mode='pickle',verbose=False,**NATIVE,chunk_size=500,online_transform=transforms.ToTensor(),label_transform=transforms.Select('label'))
            self.assertEqual(len(dataset),12);signal,label=dataset[0];self.assertEqual(label,0)
            np.testing.assert_allclose(signal.numpy(),source[:,:500]*1e-6,atol=1e-11)
            view=copy.copy(dataset);view.info=view.info.iloc[:1].copy();self.assertEqual(len(dataset),12)
            again=ADFTDRSDataset(root_path=str(root),io_path=str(cache),io_mode='pickle',verbose=False,**NATIVE,chunk_size=500)
            np.testing.assert_array_equal(again[0][0],source[:,:500].astype(np.float64)*1e-6)
            memory=Path(directory)/'unused'
            data=ADFTDRSDataset(root_path=str(root),io_path=str(memory),io_mode='memory',verbose=False,**NATIVE,chunk_size=500)
            self.assertEqual(len(data),12);self.assertFalse(memory.exists());self.assertIn('preprocessed',repr(data))

    def test_global_rng_warnings_and_constant_graph_order(self):
        state=np.random.get_state();filters=list(warnings.filters)
        import torcheeg.datasets.module.clinical._adftd as module
        importlib.reload(module)
        self.assertEqual(warnings.filters,filters)
        one=ADFTDRSDataset.fake_record();two=ADFTDRSDataset.fake_record()
        np.testing.assert_array_equal(one['segments'][0]['eeg'],two['segments'][0]['eeg'])
        for left,right in zip(state,np.random.get_state()):np.testing.assert_equal(left,right)
        self.assertEqual(ADFTD_RS_CHANNEL_LIST,CHANNELS)
        self.assertEqual(list(ADFTD_RS_CHANNEL_LOCATION_DICT),CHANNELS)
        self.assertEqual(np.asarray(ADFTD_RS_ADJACENCY_MATRIX).shape,(19,19))


if __name__=='__main__':unittest.main()
