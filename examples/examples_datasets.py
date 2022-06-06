import random
from torcheeg.datasets import DEAPDataset, DREAMERDataset, SEEDDataset
from torcheeg import transforms
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.datasets.constants.emotion_recognition.dreamer import \
    DREAMER_CHANNEL_LOCATION_DICT
from torcheeg.datasets.constants.emotion_recognition.seed import \
    SEED_CHANNEL_LOCATION_DICT

deap = DEAPDataset(
    io_path=
    f'./tmp_out/deap_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}',
    root_path='./tmp_in/data_preprocessed_python',
    offline_transform=transforms.Compose([
        transforms.BandDifferentialEntropy(),
        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
    ]),
    online_transform=transforms.ToTensor(),
    label_transform=transforms.Compose([
        transforms.Select('valence'),
        transforms.Binary(5.0),
    ]))

dreamer = DREAMERDataset(
    io_path=
    f'./tmp_out/dreamer_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}',
    mat_path='./tmp_in/DREAMER.mat',
    offline_transform=transforms.Compose([
        transforms.BandDifferentialEntropy(),
        transforms.ToGrid(DREAMER_CHANNEL_LOCATION_DICT)
    ]),
    online_transform=transforms.ToTensor(),
    label_transform=transforms.Compose([
        transforms.Select('valence'),
        transforms.Binary(3.0),
    ]))

seed = SEEDDataset(
    io_path=
    f'./tmp_out/seed_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}',
    root_path='./tmp_in/Preprocessed_EEG',
    offline_transform=transforms.Compose([
        transforms.BandDifferentialEntropy(),
        transforms.ToGrid(SEED_CHANNEL_LOCATION_DICT)
    ]),
    online_transform=transforms.ToTensor(),
    label_transform=transforms.Compose([
        transforms.Select('emotion'),
        transforms.Lambda(lambda x: x + 1),
    ]))
