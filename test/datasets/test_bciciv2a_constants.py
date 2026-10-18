import copy
import unittest

import numpy as np
import torch

from torcheeg import transforms
from torcheeg.datasets import constants
from torcheeg.datasets.constants.standard_1020 import STANDARD_1020_CHANNEL_LOCATION_DICT
from torcheeg.datasets.constants.utils import format_adj_list_from_location_list


class TestGridAdjacency(unittest.TestCase):
    def test_eight_neighbors_include_diagonals_without_gaps_or_self_loops(self):
        grid = [['A', '-', 'D'], ['B', 'C', '-']]
        self.assertEqual(format_adj_list_from_location_list(grid),
                         {'A': ['B', 'C'], 'D': ['C'], 'B': ['A', 'C'], 'C': ['A', 'D', 'B']})

    def test_empty_single_and_placeholder_only_grids(self):
        for grid in [[], [[]], np.empty((0, 3)), [['-', '-']]]:
            with self.subTest(grid=grid):
                self.assertEqual(format_adj_list_from_location_list(grid), {})
        self.assertEqual(format_adj_list_from_location_list([['C3']]), {'C3': []})

    def test_rows_columns_and_boundaries_do_not_wrap_or_bridge_empty_cells(self):
        self.assertEqual(format_adj_list_from_location_list([['A', 'B', 'C']]),
                         {'A': ['B'], 'B': ['A', 'C'], 'C': ['B']})
        self.assertEqual(format_adj_list_from_location_list([['A'], ['-'], ['C']]), {'A': [], 'C': []})

    def test_invalid_shapes_cells_and_duplicate_names_raise(self):
        grids = [[['A'], ['B', 'C']], [['A', 'A']], [['']], [[None]], [[1]],
                 'A', ['A', 'B'], None, np.empty((0, 1, 2))]
        for grid in grids:
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                format_adj_list_from_location_list(grid)

    def test_lists_and_arrays_are_not_mutated_and_results_are_independent(self):
        for grid in [[['A', 'B']], np.array([['A', 'B']])]:
            original = copy.deepcopy(grid)
            result = format_adj_list_from_location_list(grid)
            result['A'].append('C')
            np.testing.assert_array_equal(grid, original)
            self.assertEqual(format_adj_list_from_location_list(grid), {'A': ['B'], 'B': ['A']})


class TestBCICIV2aConstants(unittest.TestCase):
    def test_original_alias_order_and_grid_locations_are_unchanged(self):
        aliases = ['FZ', '0', '1', '2', '3', '4', '5', 'C3', '6', 'CZ', '7', 'C4',
                   '8', '9', '10', '11', '12', '13', '14', 'PZ', '15', '16']
        grid = [['-', '-', '-', 'FZ', '-', '-', '-'],
                ['-', '0', '1', '2', '3', '4', '-'],
                ['5', 'C3', '6', 'CZ', '7', 'C4', '8'],
                ['-', '9', '10', '11', '12', '13', '-'],
                ['-', '-', '14', 'PZ', '15', '-', '-'],
                ['-', '-', '-', '16', '-', '-', '-']]
        self.assertEqual(constants.BCICIV2A_CHANNEL_LIST, aliases)
        self.assertEqual(constants.BCICIV2A_LOCATION_LIST, grid)
        expected = {name: [row, column] for row, values in enumerate(grid)
                    for column, name in enumerate(values) if name != '-'}
        self.assertEqual(constants.BCICIV2A_LOCATION_DICT, expected)
        self.assertEqual(list(constants.BCICIV2A_LOCATION_DICT), aliases)

    def test_physical_names_align_with_all_source_columns(self):
        expected = ['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2',
                    'C4', 'C6', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P1', 'PZ', 'P2', 'POZ']
        self.assertEqual(constants.BCICIV2A_STANDARD_CHANNEL_LIST, expected)
        self.assertTrue(all(name in STANDARD_1020_CHANNEL_LOCATION_DICT for name in expected))
        self.assertEqual(len(set(expected)), 22)

    def test_grid_graph_matches_pairwise_cell_distance_for_every_edge(self):
        aliases = constants.BCICIV2A_CHANNEL_LIST
        positions = np.array([constants.BCICIV2A_LOCATION_DICT[name] for name in aliases])
        distances = np.abs(positions[:, None] - positions[None, :]).max(-1)
        expected = (distances == 1).astype(float)
        np.testing.assert_array_equal(constants.BCICIV2A_ADJACENCY_MATRIX, expected)
        self.assertEqual(constants.BCICIV2A_ADJACENCY_LIST['FZ'], ['1', '2', '3'])
        self.assertEqual(len(constants.BCICIV2A_ADJACENCY_LIST['CZ']), 8)
        np.testing.assert_array_equal(np.diag(expected), 0)
        self.assertTrue(np.all(expected.sum(0) > 0))

    def test_standard_graph_uses_physical_coordinates_and_preserves_global_links(self):
        names = constants.BCICIV2A_STANDARD_CHANNEL_LIST
        coordinates = np.array([STANDARD_1020_CHANNEL_LOCATION_DICT[name] for name in names])
        squared = ((coordinates[:, None] - coordinates[None, :]) ** 2).sum(-1)
        expected = np.minimum(1., .00035 / (squared + 1e-6))
        expected[names.index('C5'), names.index('C6')] -= 1.
        np.testing.assert_allclose(constants.BCICIV2A_STANDARD_ADJACENCY_MATRIX, expected, rtol=1e-14, atol=1e-14)
        self.assertTrue(np.isfinite(expected).all())
        self.assertTrue(np.all(np.count_nonzero(expected, axis=0) == 22))
        np.testing.assert_array_equal(np.diag(expected), 1)
        self.assertLess(expected[names.index('C5'), names.index('C6')], 0)
        self.assertGreater(expected[names.index('C6'), names.index('C5')], 0)

    def test_region_indices_cover_all_electrodes_and_match_known_groups(self):
        self.assertEqual(constants.BCICIV2A_GENERAL_REGION_LIST,
                         [[0], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12],
                          [13, 14, 15, 16, 17], [18, 19, 20], [21]])
        for groups in [constants.BCICIV2A_GENERAL_REGION_LIST, constants.BCICIV2A_FRONTAL_REGION_LIST,
                       constants.BCICIV2A_HEMISPHERE_REGION_LIST]:
            self.assertEqual(sorted(index for group in groups for index in group), list(range(22)))
        self.assertIn([0, 3, 9, 15, 19, 21], constants.BCICIV2A_HEMISPHERE_REGION_LIST)
        self.assertIn([1, 2], constants.BCICIV2A_FRONTAL_REGION_LIST)
        self.assertIn([4, 5], constants.BCICIV2A_FRONTAL_REGION_LIST)

    def test_original_grid_transform_preserves_channel_values_and_roundtrip(self):
        eeg = np.arange(22 * 8, dtype=float).reshape(22, 8)
        transform = transforms.ToGrid(constants.BCICIV2A_LOCATION_DICT)
        grid = transform(eeg=eeg)['eeg']
        self.assertEqual(grid.shape, (8, 6, 7))
        for index, name in enumerate(constants.BCICIV2A_CHANNEL_LIST):
            row, column = constants.BCICIV2A_LOCATION_DICT[name]
            np.testing.assert_array_equal(grid[:, row, column], eeg[index])
        np.testing.assert_array_equal(transform.reverse(eeg=grid)['eeg'], eeg)

    def test_graph_transform_keeps_node_order_edge_indices_and_weights(self):
        eeg = np.arange(22 * 8, dtype=np.float32).reshape(22, 8)
        for adjacency in [constants.BCICIV2A_ADJACENCY_MATRIX, constants.BCICIV2A_STANDARD_ADJACENCY_MATRIX]:
            with self.subTest(standard=adjacency is constants.BCICIV2A_STANDARD_ADJACENCY_MATRIX):
                graph = transforms.ToG(adjacency, add_self_loop=False)(eeg=eeg)['eeg']
                np.testing.assert_array_equal(graph.x.numpy(), eeg)
                dense = torch.sparse_coo_tensor(graph.edge_index, graph.edge_weight, (22, 22)).to_dense()
                torch.testing.assert_close(dense, torch.tensor(adjacency), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
