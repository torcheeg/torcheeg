import os
import random
import shutil
import unittest

from torcheeg import transforms
from torcheeg.datasets import BCICIV2aDataset
from torcheeg.datasets import StrokePatientsMIDataset,StrokePatientsMIProcessedDataset


class TestMotorImagery(unittest.TestCase):
    def setUp(self):
        if os.path.exists('./tmp_out/'):
            shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_bciciv_2a_dataset(self):
        io_path = f'./tmp_out/bciciv_2a_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/BCICIV_2a_mat'

        dataset = BCICIV2aDataset(
            io_path=io_path,
            root_path=root_path,
            online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('label'),
                                  transforms.Lambda(lambda x: x - 1)
                              ]),
            num_worker=4)

        self.assertEqual(len(dataset), 5184)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (1, 22, 1750))
        last_item = dataset[5183]
        self.assertEqual(last_item[0].shape, (1, 22, 1750))

        io_path = f'./tmp_out/bciciv_2a_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/BCICIV_2a_mat'

        dataset = BCICIV2aDataset(
            io_path=io_path,
            root_path=root_path,
            offset=1.5 * 250,
            chunk_size=4.5 * 250,
            online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('label'),
                                  transforms.Lambda(lambda x: x - 1)
                              ]),
            num_worker=4)

        self.assertEqual(len(dataset), 5184)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (1, 22, 1125))
        last_item = dataset[5183]
        self.assertEqual(last_item[0].shape, (1, 22, 1125))
    
    def test_stroke_patients_mi_dataset(self):
        io_path = f'./tmp_out/bciciv_2a_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/BCICIV_2a_mat'

        dataset = StrokePatientsMIDataset(root_path=root_path,
                                    chunk_size=500,  # 1 second
                                    overlap = 0,
                                    io_path= io_path,    
                                    num_worker=32,
                                    online_transform=transforms.BaselineCorrection()
                                    )
       

        self.assertEqual(len(dataset),8000)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (30, 500))
        last_item = dataset[7999]
        self.assertEqual(last_item[0].shape, (30,500))

        io_path = f'./tmp_out/bciciv_2a_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        dataset = StrokePatientsMIProcessedDataset(root_path=root_path,
                                    chunk_size=500, 
                                    overlap = 0,
                                    io_path= io_path,
                                    num_worker=32)
        self.assertEqual(len(dataset),8000)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (30, 500))
        last_item = dataset[7999]
        self.assertEqual(last_item[0].shape, (30,500))

if __name__ == '__main__':
    unittest.main()
