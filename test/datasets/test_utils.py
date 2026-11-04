import inspect
import unittest

import numpy as np

from torcheeg import datasets, transforms
from torcheeg.datasets.utils import get_locations


class TestDatasetMetadata(unittest.TestCase):

    def test_montages_support_grid_and_graph_transforms(self):
        from torcheeg.datasets.utils import get_adjacency, get_locations
        for cls in (datasets.AVEDDataset, datasets.AVEDProcessedDataset, datasets.DTUDataset, datasets.DTUProcessedDataset, datasets.KULDataset, datasets.KULProcessedDataset, datasets.AMIGOSDataset, datasets.BCI2022Dataset, datasets.DEAPDataset, datasets.DREAMERDataset, datasets.FACEDDataset, datasets.FACEDFeatureDataset, datasets.MAHNOBDataset, datasets.MPEDFeatureDataset, datasets.SEEDDataset, datasets.SEEDFeatureDataset, datasets.SEEDIVDataset, datasets.SEEDIVFeatureDataset, datasets.SEEDVDataset, datasets.SEEDVFeatureDataset, datasets.BCICIV2aDataset, datasets.StrokePatientsMIDataset, datasets.StrokePatientsMIProcessedDataset, datasets.M3CVDataset, datasets.HMCDataset, datasets.ISRUCDataset, datasets.P2018Dataset, datasets.SleepEDFxDataset, datasets.SanDiegoSSVEPDataset, datasets.TSUBenckmarkDataset):
            with self.subTest(dataset=cls.__name__):
                channels = cls.channels
                self.assertEqual(len(set(channels)), len(channels))
                try:
                    layout = cls.grid
                except NotImplementedError:
                    continue
                locations = get_locations(channels, layout)
                self.assertEqual(set(locations), set(channels))
                sample = np.arange(len(channels) * 8).reshape(len(channels), 8)
                grid = transforms.ToGrid(locations)(sample)
                for index, channel in enumerate(channels):
                    row, column = locations[channel]
                    np.testing.assert_array_equal(
                        grid[:, row, column], sample[index])
                try:
                    adjacency = cls.adjacency
                except NotImplementedError:
                    continue
                matrix = np.asarray(get_adjacency(channels, adjacency))
                self.assertEqual(matrix.shape, (len(channels), len(channels)))

    def test_changed_channel_order_invalidates_cache_fingerprint(self):
        from unittest.mock import patch

        from torcheeg.datasets.utils import compute_fingerprint
        cls = datasets.SEEDData
        before = compute_fingerprint(cls, {}, [], 'unused')
        with patch.object(cls, 'channels', cls.channels[::-1]):
            after = compute_fingerprint(cls, {}, [], 'unused')
        self.assertNotEqual(before, after)
        self.assertEqual(before, compute_fingerprint(cls, {}, [], 'unused'))

    def test_auditory_montages_match_verified_headers(self):
        self.assertEqual(datasets.DTUDataset.channels[47], 'CZ')
        self.assertEqual(datasets.KULDataset.channels[47], 'CZ')
        self.assertEqual(
            datasets.DTUProcessedDataset.available_channels[-2:], ['EXG1', 'EXG2'])
        self.assertEqual(len(datasets.AVEDDataset.channels), 32)
        self.assertTrue({'O1', 'OZ', 'O2'} <= set(
            datasets.AVEDDataset.channels))
        self.assertFalse({'HEOL', 'HEOR', 'VEOU', 'VEOL'} &
                         set(datasets.AVEDDataset.channels))

    def test_sleep_channels_follow_reader_defaults(self):
        for cls in (datasets.HMCDataset, datasets.ISRUCDataset, datasets.P2018Dataset, datasets.SleepEDFxDataset):
            with self.subTest(dataset=cls.__name__):
                self.assertEqual(inspect.signature(
                    cls.__init__).parameters['channels'].default, cls.channels)

    def test_grid_adjacency_ignores_empty_cells_and_includes_diagonals(self):
        from torcheeg.datasets.utils import get_grid_adjacency
        adjacency = get_grid_adjacency([['A', '-'], ['B', 'C']])
        self.assertEqual({key: set(value) for key, value in adjacency.items()}, {
                         'A': {'B', 'C'}, 'B': {'A', 'C'}, 'C': {'A', 'B'}})


class TestGrid(unittest.TestCase):

    def test_location_helper_omits_channels_absent_from_grid(self):
        channels = ['C4', 'Cz', 'C3']
        grid = [['C3', '-', 'C4']]
        mapping = get_locations(channels, grid)
        self.assertEqual(mapping, {'C4': [0, 2], 'C3': [0, 0]})
        self.assertEqual(list(mapping), ['C4', 'C3'])
        self.assertEqual(get_locations(['Cz'], grid), {})
        self.assertEqual(channels, ['C4', 'Cz', 'C3'])


if __name__ == '__main__':
    unittest.main()
