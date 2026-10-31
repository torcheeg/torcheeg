import os
import unittest

import numpy as np

from torcheeg.datasets import (AMIGOSDataset, BCI2022Dataset, DEAPDataset,
                               DREAMERDataset, FACEDDataset,
                               FACEDFeatureDataset, MAHNOBDataset,
                               MPEDFeatureDataset, SEEDDataset,
                               SEEDFeatureDataset, SEEDIVDataset,
                               SEEDIVFeatureDataset, SEEDVDataset,
                               SEEDVFeatureDataset)


class TestEmotionRecognitionDataset(unittest.TestCase):

    def test_faced_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(FACEDDataset.fake_record(**kwargs))
        item = next(FACEDDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (30, 250))

    def test_seed_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(SEEDDataset.fake_record(**kwargs))
        item = next(SEEDDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (62, 200))

    def test_faced_feature_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(FACEDFeatureDataset.fake_record(**kwargs))
        item = next(FACEDFeatureDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (30, 5))

    def test_seed_v_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(SEEDVDataset.fake_record(**kwargs))
        item = next(SEEDVDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (62, 800))
        self.assertEqual(item['info']['date'], 20180507)
        self.assertIsInstance(item['info']['date'], int)

    def test_seed_v_metadata_from_record_names(self):
        samples = np.zeros((62, 40000))
        cases = [
            ('10_1_20180507.cnt', 10, 1, 20180507, 4),
            (os.path.join('nested', '01_2_20180607.cnt'), 1, 2, 20180607, 2),
            ('2_3_20180707.CNT', 2, 3, 20180707, 2),
        ]
        for record, subject_id, session_id, date, emotion in cases:
            with self.subTest(record=record):
                item = next(SEEDVDataset.process_record(record, samples))
                info = item['info']
                for name, expected in (('subject_id', subject_id),
                                       ('session_id', session_id), ('date', date)):
                    self.assertEqual(info[name], expected)
                    self.assertIsInstance(info[name], int)
                self.assertEqual(info['emotion'], emotion)
                self.assertEqual(
                    info['clip_id'], os.path.basename(record) + '_0')
                self.assertEqual(item['eeg'].shape, (62, 800))

    def test_seed_v_feature_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(SEEDVFeatureDataset.fake_record(**kwargs))
        item = next(SEEDVFeatureDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (62, 5))

    def test_mped_feature_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(MPEDFeatureDataset.fake_record(**kwargs))
        item = next(MPEDFeatureDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (62, 23))

    def test_mahnob_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(MAHNOBDataset.fake_record(**kwargs))
        item = next(MAHNOBDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (32, 128))

    def test_amigos_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(AMIGOSDataset.fake_record(**kwargs))
        item = next(AMIGOSDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (14, 128))

    def test_deap_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(DEAPDataset.fake_record(**kwargs))
        item = next(DEAPDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (32, 128))

    def test_dreamer_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(DREAMERDataset.fake_record(**kwargs))
        item = next(DREAMERDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (14, 128))

    def test_seed_feature_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(SEEDFeatureDataset.fake_record(**kwargs))
        item = next(SEEDFeatureDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (62, 5))

    def test_seed_iv_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(SEEDIVDataset.fake_record(**kwargs))
        item = next(SEEDIVDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (62, 800))

    def test_seed_iv_feature_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(SEEDIVFeatureDataset.fake_record(**kwargs))
        item = next(SEEDIVFeatureDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (62, 5))

    def test_bci2022(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(BCI2022Dataset.fake_record(**kwargs))
        item = next(BCI2022Dataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (30, 250))


if __name__ == '__main__':
    unittest.main()
