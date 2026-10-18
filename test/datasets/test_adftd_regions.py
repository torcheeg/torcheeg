import copy
import importlib
import unittest

import numpy as np

from torcheeg.datasets.constants import region_1020
from torcheeg.datasets.constants.clinical import adftd_rs, adftd_ps
from torcheeg.datasets.constants.standard_1005 import STANDARD_1005_CHANNEL_LOCATION_DICT


CHANNELS = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','CZ','PZ']
EXPECTED = {
    'GENERAL': [[0,1],[10,2,16,3,11],[4,17,5],[14,6,18,7,15],[8,9],[12],[13]],
    'FRONTAL': [[10,2],[3,11],[0],[1],[16],[4,17,5],[14,6,18,7,15],[8,9],[12],[13]],
    'HEMISPHERE': [[10,2],[3,11],[0],[1],[16,17,18],[4],[5],[14,6],[7,15],[8],[9],[12],[13]],
}
MODULES = [(adftd_rs,'ADFTD_RS'),(adftd_ps,'ADFTD_PS')]


class TestADFTDRegions(unittest.TestCase):
    def test_every_signal_column_occurs_once_in_each_region_family(self):
        for module,prefix in MODULES:
            for family in EXPECTED:
                with self.subTest(dataset=prefix,family=family):
                    regions=getattr(module,prefix+'_'+family+'_REGION_LIST')
                    indices=[index for region in regions for index in region]
                    self.assertEqual(sorted(indices),list(range(19)))

    def test_exact_anatomical_groups_keep_legacy_source_column_indices(self):
        for module,prefix in MODULES:
            for family,expected in EXPECTED.items():
                with self.subTest(dataset=prefix,family=family):
                    self.assertEqual(getattr(module,prefix+'_'+family+'_REGION_LIST'),expected)

    def test_temporal_signals_contribute_to_actual_regional_pooling(self):
        source=np.zeros((19,7))
        for index in range(12,16):source[index]=np.arange(7)+(index-11)*10
        for module,prefix in MODULES:
            for family,expected in EXPECTED.items():
                with self.subTest(dataset=prefix,family=family):
                    regions=getattr(module,prefix+'_'+family+'_REGION_LIST')
                    actual=np.stack([source[indices].mean(0) for indices in regions])
                    reference=np.stack([source[indices].mean(0) for indices in expected])
                    np.testing.assert_array_equal(actual,reference)
                    reconstructed=sum((source[indices].sum(0) for indices in regions),np.zeros(7))
                    np.testing.assert_array_equal(reconstructed,source.sum(0))

    def test_legacy_names_grids_and_graphs_retain_their_source_order(self):
        for module,prefix in MODULES:
            self.assertEqual(getattr(module,prefix+'_CHANNEL_LIST'),CHANNELS)
            positions=getattr(module,prefix+'_CHANNEL_LOCATION_DICT')
            grid=getattr(module,prefix+'_LOCATION_LIST')
            for name in CHANNELS:
                row,column=positions[name];self.assertEqual(grid[row][column],name)
            adjacency=np.asarray(getattr(module,prefix+'_ADJACENCY_MATRIX'))
            expected=np.zeros((19,19))
            for name,neighbors in getattr(module,prefix+'_ADJACENCY_LIST').items():
                for neighbor in neighbors:expected[CHANNELS.index(name),CHANNELS.index(neighbor)]=1
            np.testing.assert_array_equal(adjacency,expected)
            weighted=np.asarray(getattr(module,prefix+'_STANDARD_ADJACENCY_MATRIX'))
            self.assertEqual(weighted.shape,(19,19));self.assertTrue(np.isfinite(weighted).all())
            np.testing.assert_array_equal(weighted.diagonal(),np.ones(19))
        for legacy,modern in [('T3','T7'),('T4','T8'),('T5','P7'),('T6','P8')]:
            np.testing.assert_array_equal(STANDARD_1005_CHANNEL_LOCATION_DICT[legacy],STANDARD_1005_CHANNEL_LOCATION_DICT[modern])

    def test_import_does_not_rewrite_global_templates_or_source_names(self):
        originals={family:copy.deepcopy(getattr(region_1020,family+'_REGION_LIST')) for family in EXPECTED}
        for module,prefix in MODULES:
            importlib.reload(module)
            self.assertEqual(getattr(module,prefix+'_CHANNEL_LIST'),CHANNELS)
        for family,value in originals.items():self.assertEqual(getattr(region_1020,family+'_REGION_LIST'),value)


if __name__=='__main__':unittest.main()
