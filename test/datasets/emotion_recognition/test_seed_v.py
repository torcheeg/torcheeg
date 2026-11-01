import os
import unittest

import numpy as np

from torcheeg.datasets import SEEDVDataset


class TestSeedV(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
