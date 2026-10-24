import unittest

import numpy as np

from torcheeg.datasets import constants
from torcheeg.transforms import Compose, ToGrid


class TestGrid(unittest.TestCase):
    def test_non_square_grid_roundtrip_matches_channel_order(self):
        mapping = {'Cz': (2, 1), 'Fp1': (0, 0), 'Pz': (4, 2)}
        eeg = np.arange(21, dtype=float).reshape(3, 7)
        transform = ToGrid(mapping)
        grid = transform(eeg=eeg)['eeg']
        self.assertEqual(grid.shape, (7, 5, 3))
        np.testing.assert_array_equal(grid[:, 2, 1], eeg[0])
        np.testing.assert_array_equal(grid[:, 0, 0], eeg[1])
        np.testing.assert_array_equal(grid[:, 4, 2], eeg[2])
        np.testing.assert_array_equal(transform.reverse(eeg=grid)['eeg'], eeg)
        self.assertTrue((grid[:, 1, 1] == 0).all())

    def test_missing_electrodes_must_be_removed_from_mapping_and_input(self):
        eeg = np.arange(15).reshape(3, 5)  # C3, Cz, C4
        with self.assertRaisesRegex(ValueError, 'both the mapping and the EEG input'):
            ToGrid({'C3': (0, 0), 'Cz': None, 'C4': (0, 2)})
        transform = ToGrid({'C3': (0, 0), 'C4': (0, 2)})
        with self.assertRaisesRegex(ValueError, 'one row per channel mapping'):
            transform(eeg=eeg)
        selected = eeg[[0, 2]]
        grid = transform(eeg=selected)['eeg']
        np.testing.assert_array_equal(grid[:, 0, 0], eeg[0])
        np.testing.assert_array_equal(grid[:, 0, 2], eeg[2])
        recovered = transform.reverse(eeg=grid)['eeg']
        self.assertEqual(recovered.shape, (2, 5))
        np.testing.assert_array_equal(recovered, selected)

    def test_empty_grid_cells_do_not_change_recovered_electrodes(self):
        transform = ToGrid({'first': (0, 0), 'last': (2, 2)})
        eeg = np.arange(10).reshape(2, 5)
        grid = transform(eeg=eeg)['eeg']
        grid[:, 1, 1] = 999
        np.testing.assert_array_equal(transform.reverse(eeg=grid)['eeg'], eeg)

    def test_baseline_and_compose_contracts(self):
        transform = ToGrid({'a': (0, 0), 'b': (1, 1)}, apply_to_baseline=True)
        eeg, baseline = np.arange(10).reshape(2, 5), np.ones((2, 5))
        result = Compose([transform])(eeg=eeg, baseline=baseline, y=7)
        np.testing.assert_array_equal(transform.reverse(eeg=result['eeg'])['eeg'], eeg)
        np.testing.assert_array_equal(transform.reverse(eeg=result['baseline'])['eeg'], baseline)
        self.assertEqual(result['y'], 7)
        default = ToGrid({'a': (0, 0), 'b': (1, 1)})(eeg=eeg, baseline=baseline)
        self.assertIs(default['baseline'], baseline)

    def test_mapping_is_copied_without_changing_insertion_order(self):
        mapping = {'last': [1, 1], 'first': [0, 0]}
        transform = ToGrid(mapping)
        mapping['last'][0] = 10
        mapping['extra'] = [2, 2]
        self.assertEqual(list(transform.channel_location_dict), ['last', 'first'])
        eeg = np.arange(10).reshape(2, 5)
        np.testing.assert_array_equal(transform.reverse(eeg=transform(eeg=eeg)['eeg'])['eeg'], eeg)

    def test_numpy_integer_grid_positions_are_valid(self):
        transform = ToGrid({'a': np.array([1, 2], dtype=np.int64)})
        self.assertEqual(transform(eeg=np.ones((1, 5)))['eeg'].shape, (5, 2, 3))

    def test_3d_montages_raise_an_actionable_grid_error(self):
        with self.assertRaisesRegex(ValueError, '2D integer.*3D standard-montage'):
            ToGrid({'Fp1': (0.1, 0.2, 0.3)})

    def test_invalid_positions_fail_before_conversion(self):
        mappings = [{}, {'a': None}, {'a': (-1, 0)}, {'a': (0.1, 1)},
                    {'a': (True, 0)}, {'a': 'ab'}, {'a': 1},
                    {'a': (0, 0), 'b': (0, 0)}]
        for mapping in mappings:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                ToGrid(mapping)

    def test_forward_and_reverse_shapes_must_match_the_mapping(self):
        transform = ToGrid({'a': (0, 1), 'b': (2, 0)})
        for eeg in (np.ones(5), np.ones((1, 5)), np.ones((3, 5)), np.ones((1, 2, 5))):
            with self.subTest(shape=eeg.shape), self.assertRaises(ValueError):
                transform(eeg=eeg)
        for grid in (np.ones((3, 2)), np.ones((5, 2, 3)), np.ones((5, 3, 3))):
            with self.subTest(shape=grid.shape), self.assertRaises(ValueError):
                transform.reverse(eeg=grid)

    def test_input_arrays_are_unchanged(self):
        transform = ToGrid({'a': (0, 1), 'b': (2, 0)})
        eeg = np.arange(10).reshape(2, 5)
        original = eeg.copy()
        grid = transform(eeg=eeg)['eeg']
        grid_original = grid.copy()
        recovered = transform.reverse(eeg=grid)['eeg']
        recovered[:] = -1
        np.testing.assert_array_equal(eeg, original)
        np.testing.assert_array_equal(grid, grid_original)

    def test_public_grids_with_missing_positions_require_channel_selection(self):
        checked = 0
        for name in dir(constants):
            if not name.endswith('CHANNEL_LOCATION_DICT'):
                continue
            mapping = getattr(constants, name)
            positions = [location for location in mapping.values() if location is not None]
            if not positions or not all(len(location) == 2 for location in positions):
                continue
            with self.subTest(name=name):
                indices = [index for index, location in enumerate(mapping.values())
                           if location is not None]
                if len(indices) != len(mapping):
                    with self.assertRaisesRegex(ValueError, 'None coordinates'):
                        ToGrid(mapping)
                selected_mapping = {name: location for name, location in mapping.items()
                                    if location is not None}
                transform = ToGrid(selected_mapping)
                eeg = np.arange(len(mapping) * 3).reshape(len(mapping), 3)
                selected = eeg[indices]
                recovered = transform.reverse(eeg=transform(eeg=selected)['eeg'])['eeg']
                np.testing.assert_array_equal(recovered, selected)
                checked += 1
        self.assertGreater(checked, 5)


if __name__ == '__main__':
    unittest.main()
