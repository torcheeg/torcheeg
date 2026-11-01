import unittest

from torcheeg.datasets import (StrokePatientsMIDataset,
                               StrokePatientsMIProcessedDataset)


class TestStrokes(unittest.TestCase):

    def test_stroke_patients_mi_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(StrokePatientsMIDataset.fake_record(**kwargs))
        item = next(StrokePatientsMIDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (33, 1000))

        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(StrokePatientsMIProcessedDataset.fake_record(**kwargs))
        item = next(StrokePatientsMIProcessedDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (30, 1000))


if __name__ == '__main__':
    unittest.main()
