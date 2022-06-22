import logging
import time

from torcheeg import transforms
from torcheeg.datasets import DEAPDataset, SEEDDataset
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.datasets.constants.emotion_recognition.seed import \
    SEED_ADJACENCY_MATRIX

logger = logging.getLogger('dataset_runtime')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('./tmp_out/dataset_runtime.log')
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def deap_dataset_runtime(num_worker):
    start = time.time()
    dataset = DEAPDataset(io_path=f'./tmp_out/deap_{num_worker}',
                          root_path='./tmp_in/data_preprocessed_python',
                          offline_transform=transforms.Compose([
                              transforms.BandDifferentialEntropy(apply_to_baseline=True),
                              transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True)
                          ]),
                          online_transform=transforms.Compose([transforms.BaselineRemoval(),
                                                               transforms.ToTensor()]),
                          label_transform=transforms.Compose([
                              transforms.Select('valence'),
                              transforms.Binary(5.0),
                          ]),
                          num_worker=num_worker)
    end = time.time()
    logger.info(dataset)
    logger.info(f'Runtime: {end - start}s')


def seed_dataset_runtime(num_worker):
    start = time.time()
    dataset = SEEDDataset(io_path=f'./tmp_out/seed_{num_worker}',
                          root_path='./tmp_in/Preprocessed_EEG',
                          offline_transform=transforms.BandDifferentialEntropy(),
                          online_transform=transforms.ToG(SEED_ADJACENCY_MATRIX),
                          label_transform=transforms.Compose([
                              transforms.Select('emotion'),
                              transforms.Lambda(lambda x: x + 1),
                          ]),
                          num_worker=num_worker)
    end = time.time()
    logger.info(dataset)
    logger.info(f'Runtime: {end - start}s')

if __name__ == "__main__":

    for n in [2, 4, 8, 16, 32, 64]:
        deap_dataset_runtime(n)

    for n in [2, 4, 8, 16, 32, 64]:
        seed_dataset_runtime(n)
