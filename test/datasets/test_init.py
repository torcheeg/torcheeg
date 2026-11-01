import importlib
import unittest


class TestDatasetsExports(unittest.TestCase):
    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.datasets')
        expected = {'AMIGOSDataset': 'torcheeg.datasets.emotion_recognition.amigos',
                    'AVEDDataset': 'torcheeg.datasets.auditory_attention.aved',
                    'AVEDProcessedDataset': 'torcheeg.datasets.auditory_attention.aved_processed',
                    'BCI2022Dataset': 'torcheeg.datasets.emotion_recognition.bci2022',
                    'BCICIV2aDataset': 'torcheeg.datasets.motor_imagery.bciciv_2a',
                    'BaseDataset': 'torcheeg.datasets.base_dataset',
                    'CSVFolderDataset': 'torcheeg.datasets.csv_folder_dataset',
                    'ConcatDataset': 'torcheeg.datasets.concat_dataset',
                    'DEAPDataset': 'torcheeg.datasets.emotion_recognition.deap',
                    'DREAMERDataset': 'torcheeg.datasets.emotion_recognition.dreamer',
                    'DTUDataset': 'torcheeg.datasets.auditory_attention.dtu',
                    'DTUProcessedDataset': 'torcheeg.datasets.auditory_attention.dtu_processed',
                    'FACEDDataset': 'torcheeg.datasets.emotion_recognition.faced',
                    'FACEDFeatureDataset': 'torcheeg.datasets.emotion_recognition.faced_feature',
                    'FolderDataset': 'torcheeg.datasets.folder_dataset',
                    'HMCDataset': 'torcheeg.datasets.sleep_stage_detection.hmc',
                    'ISRUCDataset': 'torcheeg.datasets.sleep_stage_detection.isruc',
                    'KULDataset': 'torcheeg.datasets.auditory_attention.kul',
                    'KULProcessedDataset': 'torcheeg.datasets.auditory_attention.kul_processed',
                    'M3CVDataset': 'torcheeg.datasets.personal_identification.m3cv',
                    'MAHNOBDataset': 'torcheeg.datasets.emotion_recognition.mahnob',
                    'MNERawDataset': 'torcheeg.datasets.mne_raw_dataset',
                    'MPEDFeatureDataset': 'torcheeg.datasets.emotion_recognition.mped_feature',
                    'P2018Dataset': 'torcheeg.datasets.sleep_stage_detection.p2018',
                    'PairDataset': 'torcheeg.datasets.pair_dataset',
                    'SEEDDataset': 'torcheeg.datasets.emotion_recognition.seed',
                    'SEEDFeatureDataset': 'torcheeg.datasets.emotion_recognition.seed_feature',
                    'SEEDIVDataset': 'torcheeg.datasets.emotion_recognition.seed_iv',
                    'SEEDIVFeatureDataset': 'torcheeg.datasets.emotion_recognition.seed_iv_feature',
                    'SEEDVDataset': 'torcheeg.datasets.emotion_recognition.seed_v',
                    'SEEDVFeatureDataset': 'torcheeg.datasets.emotion_recognition.seed_v_feature',
                    'SanDiegoSSVEPDataset': 'torcheeg.datasets.ssvep.sandiego_ssvep',
                    'SleepEDFxDataset': 'torcheeg.datasets.sleep_stage_detection.sleep_edfx',
                    'StrokePatientsMIDataset': 'torcheeg.datasets.motor_imagery.strokes',
                    'StrokePatientsMIProcessedDataset': 'torcheeg.datasets.motor_imagery.strokes',
                    'TSUBenckmarkDataset': 'torcheeg.datasets.ssvep.tsu_benchmark'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == "__main__":
    unittest.main()
