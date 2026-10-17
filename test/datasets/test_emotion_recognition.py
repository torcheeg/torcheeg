import ast
import inspect
import os
import tempfile
import textwrap
import unittest

import numpy as np
import pandas as pd

from torcheeg import transforms
from torcheeg.datasets import BaseDataset
from torcheeg.io import EEGSignalIO
from torcheeg.model_selection import KFoldCrossTrial

from torcheeg.datasets import (
    AMIGOSDataset, BCI2022Dataset, DEAPDataset, DREAMERDataset, FACEDDataset,
    FACEDFeatureDataset, MAHNOBDataset, MPEDFeatureDataset, SEEDDataset,
    SEEDFeatureDataset, SEEDIVDataset, SEEDIVFeatureDataset, SEEDVDataset,
    SEEDVFeatureDataset)


class TestEmotionRecognitionDataset(unittest.TestCase):

    def test_faced_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(FACEDDataset.fake_record(**kwargs))
        item = next(FACEDDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (30, 250))

    def test_faced_documented_emotion_labels(self):
        # Exercise each documented label transform against all nine classes.
        docstring = inspect.getdoc(FACEDDataset)
        examples = []
        lines = docstring.splitlines()
        for index, line in enumerate(lines):
            if line.strip() != '.. code-block:: python':
                continue
            block = []
            for code_line in lines[index + 1:]:
                if code_line and not code_line.startswith('    '):
                    break
                block.append(code_line)
            examples.append(ast.parse(textwrap.dedent('\n'.join(block))))

        label_transforms = []
        for example in examples:
            for node in ast.walk(example):
                if not isinstance(node, ast.keyword) or node.arg != 'label_transform':
                    continue
                expression = ast.Expression(body=node.value)
                label_transforms.append(eval(
                    compile(expression, '<FACEDDataset example>', 'eval'),
                    {'transforms': transforms}))

        self.assertEqual(len(label_transforms), 3)
        for label_transform in label_transforms:
            for emotion in range(9):
                with self.subTest(emotion=emotion, transform=label_transform):
                    self.assertEqual(label_transform(y={'emotion': emotion})['y'],
                                     emotion)

    def test_seed_dataset(self):
        kwargs = {
            'record': 'fake_record',
        }
        kwargs.update(SEEDDataset.fake_record(**kwargs))
        item = next(SEEDDataset.process_record(**kwargs))

        self.assertEqual(item['eeg'].shape, (62, 200))

    def test_seed_trial_ids_are_shared_across_subjects(self):
        labels = np.array([-1, 0, 1] * 5)
        records = []
        for subject_id, prefix in ((1, 'jl'), (2, 'phl')):
            samples = {f'{prefix}_eeg{trial_id}': np.full((62, 400), trial_id)
                       for trial_id in range(1, 16)}
            samples['__header__'] = 'metadata'
            items = list(SEEDDataset.process_record(
                record=f'{subject_id}_20130703.mat', samples=samples,
                labels=labels))
            self.assertEqual(len(items), 30)
            for item in items:
                info = item['info']
                self.assertIsInstance(info['trial_id'], int)
                self.assertEqual(info['subject_id'], subject_id)
                self.assertEqual(info['emotion'], labels[info['trial_id'] - 1])
                np.testing.assert_array_equal(item['eeg'],
                    np.full((62, 200), info['trial_id']))
                records.append(info)
            self.assertEqual({item['info']['trial_id'] for item in items},
                             set(range(1, 16)))

        dataset = BaseDataset.__new__(BaseDataset)
        dataset.info = pd.DataFrame(records)
        dataset.eeg_io_router = {'_record_0': EEGSignalIO(
            io_path='unused', io_mode='memory')}
        dataset.eeg_signal_paths = {}
        with tempfile.TemporaryDirectory() as directory:
            cv = KFoldCrossTrial(n_splits=5,
                                split_path=os.path.join(directory, 'split'))
            folds = list(cv.split(dataset))
        self.assertEqual(len(folds), 5)
        for train, test in folds:
            self.assertEqual(len(train), 48)
            self.assertEqual(len(test), 12)
            self.assertFalse(set(train.info['trial_id']) & set(test.info['trial_id']))
            self.assertEqual(set(train.info['subject_id']), {1, 2})
            self.assertEqual(set(test.info['subject_id']), {1, 2})

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
                self.assertEqual(info['clip_id'], os.path.basename(record) + '_0')
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
