import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd


class TestMOABBDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        stubs = {}
        for name in ('moabb', 'moabb.datasets', 'moabb.datasets.base', 'moabb.paradigms', 'moabb.paradigms.base'):
            stubs[name] = types.ModuleType(name)
        stubs['moabb.datasets.base'].BaseDataset = type('BaseDataset', (), {})
        stubs['moabb.paradigms.base'].BaseParadigm = type(
            'BaseParadigm', (), {})
        path = Path(__file__).resolve(
        ).parents[3] / 'torcheeg/datasets/moabb/moabb_dataset.py'
        spec = importlib.util.spec_from_file_location(
            'torcheeg.datasets.moabb._test_adapter', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, stubs):
            spec.loader.exec_module(module)
        cls._adapter = module.MOABBDataset

    def setUp(self):
        self._signal = np.arange(32).reshape(2, 2, 8)
        self._dataset = types.SimpleNamespace(subject_list=[1, 2], code='fake', download=Mock(
        ), get_data=Mock(return_value={1: {'session': {'run': 'raw'}}}))
        self._paradigm = types.SimpleNamespace(is_valid=Mock(return_value=True), process_raw=Mock(
            return_value=(self._signal, ['left', 'right'], pd.DataFrame({'event': [10, 20]}))))

    def test_windows_keep_labels_metadata_and_unique_keys(self):
        data = self._adapter.data_class(dataset=self._dataset, paradigm=self._paradigm, chunk_size=4,
                                        overlap=2, transform=lambda sample: {**sample, 'eeg': sample['eeg'] + 1})
        items = list(data.iter_record((1, 'session')))
        self.assertEqual(len(items), 6)
        self.assertEqual(len(
            {(item['info']['metadata']['event'], item['info']['start_at']) for item in items}), 6)
        for index, item in enumerate(items):
            roi, window = divmod(index, 3)
            start = window * 2
            np.testing.assert_array_equal(
                item['eeg'], self._signal[roi, :, start:start + 4] + 1)
            self.assertEqual(item['info']['subject_id'], 1)
            self.assertEqual(item['info']['trial_id'], 'run')
            self.assertEqual(item['info']['label'], ['left', 'right'][roi])
            self.assertEqual(item['info']['metadata'],
                             {'event': [10, 20][roi]})
            self.assertNotIn('key', item)
        self._dataset.download.assert_not_called()

    def test_default_windows_and_runs_without_selected_events(self):
        items = list(self._adapter.data_class(dataset=self._dataset,
                     paradigm=self._paradigm).process_record(file=(1, 'session')))
        self.assertEqual(len(items), 2)
        np.testing.assert_array_equal(items[0]['eeg'], self._signal[0])
        self._paradigm.process_raw.return_value = None
        self.assertEqual(list(self._adapter.data_class(
            dataset=self._dataset, paradigm=self._paradigm).process_record(file=(1, 'session'))), [])
        self.assertEqual(self._paradigm.process_raw.call_args.args[0], 'raw')

    def test_record_listing_reuses_local_fake_data_without_downloading(self):
        adapter = self._adapter.data_class(
            dataset=self._dataset, paradigm=self._paradigm)
        with tempfile.TemporaryDirectory() as directory:
            records = adapter.set_records(
                self._dataset, download_path=directory)
        self.assertEqual(records, [(1, 'session'), (2, 'session')])
        self._dataset.download.assert_not_called()

    def test_invalid_paradigm_fails_before_preprocessing_or_downloading(self):
        self._paradigm.is_valid.return_value = False
        with self.assertRaisesRegex(ValueError, 'not valid'):
            self._adapter(dataset=self._dataset, paradigm=self._paradigm,
                          download_path='unused', io_mode='memory')
        self._dataset.get_data.assert_not_called()
        self._dataset.download.assert_not_called()

    def test_memory_construction_uses_the_data_adapter(self):
        self._dataset.subject_list = [1]
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._adapter(dataset=self._dataset, paradigm=self._paradigm,
                                    download_path=directory, io_mode='memory', verbose=False)
        self.assertEqual(len(dataset), 2)
        sample = dataset[0]
        signal, info = sample['eeg'], sample['info']
        np.testing.assert_array_equal(signal, self._signal[0])
        self.assertEqual(info['label'], 'left')
        self.assertEqual(info['metadata'], {'event': 10})
        self._dataset.download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
