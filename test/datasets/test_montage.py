import unittest

import numpy as np

from torcheeg.datasets.montage import (FRONTAL_REGION_LIST,
                                       GENERAL_REGION_LIST,
                                       HEMISPHERE_REGION_LIST,
                                       STANDARD_1005_CHANNEL_LOCATION_DICT,
                                       STANDARD_1020_CHANNEL_LOCATION_DICT)
from torcheeg.datasets.utils import get_positions, get_regions


class TestMontage(unittest.TestCase):
    def test_standard_coordinates_are_finite_three_dimensional_positions(self):
        for montage in (STANDARD_1005_CHANNEL_LOCATION_DICT,
                        STANDARD_1020_CHANNEL_LOCATION_DICT):
            for channel, position in montage.items():
                with self.subTest(channel=channel):
                    self.assertEqual(np.asarray(position).shape, (3,))
                    self.assertTrue(np.isfinite(position).all())
        self.assertTrue({'FP1', 'CZ', 'O2'} <=
                        set(STANDARD_1020_CHANNEL_LOCATION_DICT))

    def test_positions_follow_requested_channel_order(self):
        positions = get_positions(['O2', 'FP1', 'CZ'])
        self.assertEqual(list(positions), ['O2', 'FP1', 'CZ'])
        for channel, position in positions.items():
            np.testing.assert_array_equal(
                position, STANDARD_1005_CHANNEL_LOCATION_DICT[channel])

    def test_region_definitions_form_valid_channel_groups(self):
        for regions in (GENERAL_REGION_LIST, FRONTAL_REGION_LIST,
                        HEMISPHERE_REGION_LIST):
            with self.subTest(regions=regions):
                self.assertTrue(all(regions))
                channels = list(dict.fromkeys(
                    c for group in regions for c in group))
                groups = get_regions(channels, regions)
                self.assertEqual(groups,
                                 [[channels.index(c) for c in group]
                                  for group in regions])


if __name__ == '__main__':
    unittest.main()
