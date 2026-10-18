import csv
import json
from pathlib import Path
import tempfile
import unittest

import mne
import numpy as np
from scipy.io import loadmat, savemat

from test.datasets.test_adftd_rs import NATIVE, participants, write_record
from torcheeg import transforms
from torcheeg.datasets import ADFTDPSDataset, ADFTDRSDataset
from torcheeg.datasets.constants import ADFTD_PS_CHANNEL_LIST, ADFTD_PS_CHANNEL_LOCATION_DICT
from torcheeg.datasets.module.clinical._adftd import CHANNELS


def write_photo_record(root, subject='sub-001', run=None, preprocessed=False, boundary=None, samples=6000):
    old,source=write_record(root,subject=subject,run=run,preprocessed=preprocessed,boundary=boundary,samples=samples)
    identity=old.stem.replace('task-eyesclosed','task-photomark')
    folder=Path(root)/('derivatives/eeglab' if preprocessed else '')/subject/'eeg';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(identity+'.set');old.rename(path)
    rawfolder=Path(root)/subject/'eeg'
    for suffix in ['_channels.tsv','_eeg.json']:
        previous=rawfolder/(old.stem[:-4]+suffix)
        previous.rename(rawfolder/(identity[:-4]+suffix))
    metadata=rawfolder/(identity+'.json')
    header=json.loads(metadata.read_text());header['TaskName']='photomark';metadata.write_text(json.dumps(header))
    channels=rawfolder/(identity[:-4]+'_channels.tsv')
    with channels.open('w',newline='') as file:
        writer=csv.writer(file,delimiter='\t');writer.writerow(['name','type','units']);writer.writerows((channel,'n/a','n/a') for channel in CHANNELS)
    # Photostimulation markers are clinical acquisition events, not diagnosis targets.
    if boundary is None:
        payload={key:value for key,value in loadmat(path,simplify_cells=True).items() if not key.startswith('__')}
        payload['event']=np.array([dict(type='PHOTO 5Hz',latency=1899.5,duration=0),dict(type='Photo/HV mark',latency=1901.,duration=0)],dtype=object)
        savemat(path,payload)
    return path,source


def rows(record, **options):
    options=dict(NATIVE,**options)
    data=ADFTDPSDataset.read_record(record,**options)
    return list(ADFTDPSDataset.process_record(record=record,**data,chunk_size=500,num_channel=options.get('num_channel',19)))


class TestADFTDPSDataset(unittest.TestCase):
    def test_actual_photo_mat_all_diagnoses_ignore_stimulus_as_target_and_unknown_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);participants(root,[('sub-001','A'),('sub-002','C'),('sub-003','F')],bom=True)
            sources={subject:write_photo_record(root,subject)[1] for subject in ['sub-001','sub-002','sub-003']}
            found=ADFTDPSDataset.set_records(root_path=str(root));self.assertEqual(len(found),3)
            for record in found:
                output=rows(record);self.assertEqual(len(output),12)
                for index,item in enumerate(output):
                    np.testing.assert_array_equal(item['eeg'],sources[record['participant_id']][:,index*500:(index+1)*500].astype(np.float64)*1e-6)
                    info=item['info'];self.assertEqual(info['task'],'photomark')
                    self.assertEqual(info['label'],{'C':0,'A':1,'F':2}[record['group']])
                    self.assertEqual(info['signal_unit'],'V');self.assertEqual(info['units_basis'],'EEGLAB_microvolt_format_convention')
                    self.assertEqual(set(json.loads(info['channel_units_metadata']).values()),{'n/a'})
                    self.assertEqual(json.loads(info['channel_names']),CHANNELS)
                self.assertEqual(len({item['info']['trial_id'] for item in output}),1)

    def test_all_runs_retained_and_wrong_task_is_never_substituted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);participants(root);write_record(root)
            with self.assertRaises(FileNotFoundError):ADFTDPSDataset.set_records(root_path=str(root))
            write_photo_record(root,run='01');write_photo_record(root,run='02')
            records=ADFTDPSDataset.set_records(root_path=str(root))
            self.assertEqual(len(records),2)
            output=[item for record in records for item in rows(record)]
            self.assertEqual(len({item['key'] for item in output}),24)
            self.assertEqual(len({item['info']['record_id'] for item in output}),2)

    def test_provider_derivatives_use_eeglab_directory_and_do_not_cross_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);participants(root);write_photo_record(root)
            path,source=write_photo_record(root,preprocessed=True,boundary=2500.5,samples=5000)
            (root/'sub-001/eeg/sub-001_task-photomark_eeg.json').write_text(json.dumps(dict(TaskName='photomark',SamplingFrequency=500,RecordingDuration=12.)))
            record=ADFTDPSDataset.set_records(root_path=str(root),preprocessed=True)[0]
            self.assertEqual(Path(record['file_path']),path)
            self.assertIn('/derivatives/eeglab/',record['file_path'])
            data=ADFTDPSDataset.read_record(record,preprocessed=True,**NATIVE)
            output=list(ADFTDPSDataset.process_record(record=record,**data,chunk_size=1500))
            self.assertEqual(len(output),2)
            for item,(first,last) in zip(output,[(0,1500),(2500,4000)]):
                np.testing.assert_array_equal(item['eeg'],source[:,first:last].astype(np.float64)*1e-6)
                self.assertEqual(item['info']['source_start_seconds'],first/500)
            self.assertEqual([item['info']['segment_index'] for item in output],[0,1])
            self.assertTrue(all(item['info']['data_source']=='preprocessed' for item in output))

    def test_default_filter_and_resample_matches_independent_native_data(self):
        with tempfile.TemporaryDirectory() as directory:
            participants(directory);_,source=write_photo_record(directory)
            record=ADFTDPSDataset.set_records(root_path=str(directory))[0]
            data=ADFTDPSDataset.read_record(record)
            output=list(ADFTDPSDataset.process_record(record=record,**data))
            raw=mne.io.RawArray(source.astype(np.float64)*1e-6,mne.create_info(CHANNELS,500,'eeg'),verbose=False)
            raw.notch_filter(50.,verbose=False);raw.filter(.5,45.,verbose=False);raw.resample(200,verbose=False)
            self.assertEqual(len(output),6)
            for index,item in enumerate(output):np.testing.assert_array_equal(item['eeg'],raw.get_data()[:,index*400:(index+1)*400])

    def test_cache_transforms_and_same_cohort_id_across_rs_and_ps(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rs=root/'rs';ps=root/'ps'
            participants(rs,[('sub-001','F')]);participants(ps,[('sub-001','F')])
            write_record(rs);_,source=write_photo_record(ps)
            rsrecord=ADFTDRSDataset.set_records(root_path=str(rs))[0]
            psrecord=ADFTDPSDataset.set_records(root_path=str(ps))[0]
            self.assertEqual(rsrecord['subject_id'],psrecord['subject_id'])
            self.assertNotEqual(rsrecord['record_id'],psrecord['record_id'])
            dataset=ADFTDPSDataset(root_path=str(ps),io_path=str(root/'cache'),io_mode='pickle',verbose=False,
                                  **NATIVE,chunk_size=500,online_transform=transforms.ToTensor(),label_transform=transforms.Select('label'))
            self.assertEqual(len(dataset),12);signal,label=dataset[0];self.assertEqual(label,2)
            np.testing.assert_allclose(signal.numpy(),source[:,:500]*1e-6,atol=1e-11)
            fake=ADFTDPSDataset.fake_record();item=next(ADFTDPSDataset.process_record(**fake))
            self.assertEqual(item['info']['task'],'photomark');self.assertTrue(item['info']['record_id'].startswith('ADFTD-PS_'))

    def test_wrong_metadata_task_or_diagnosis_and_missing_files_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            participants(directory);path,_=write_photo_record(directory)
            record=ADFTDPSDataset.set_records(root_path=str(directory))[0]
            with self.assertRaises(ValueError):ADFTDPSDataset.read_record(dict(record,label=2),**NATIVE)
            header=json.loads(path.with_suffix('.json').read_text());header['TaskName']='eyesclosed';path.with_suffix('.json').write_text(json.dumps(header))
            with self.assertRaises(ValueError):ADFTDPSDataset.read_record(record,**NATIVE)
            path.unlink()
            with self.assertRaises(FileNotFoundError):ADFTDPSDataset.set_records(root_path=str(directory))

    def test_canonical_grid_constants_follow_the_published_channel_order(self):
        self.assertEqual(ADFTD_PS_CHANNEL_LIST,CHANNELS)
        self.assertEqual(list(ADFTD_PS_CHANNEL_LOCATION_DICT),CHANNELS)
        self.assertEqual(len({tuple(value) for value in ADFTD_PS_CHANNEL_LOCATION_DICT.values()}),19)


if __name__=='__main__':unittest.main()
