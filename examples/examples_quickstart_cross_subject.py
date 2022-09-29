import logging
import os
import time

from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.models import CCNN
from torcheeg.trainers import DDCTrainer

os.makedirs("./tmp_out/examples_quickstart_cross_subject", exist_ok=True)

logger = logging.getLogger('Examples of cross-subject EEG analysis')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
file_handler = logging.FileHandler(
    f"./tmp_out/examples_quickstart_cross_subject/{timeticks}.log")
logger.addHandler(console_handler)
logger.addHandler(file_handler)


class MyDDCTrainer(DDCTrainer):
    def print(self, *args, **kwargs):
        if self.is_main:
            logger.info(*args, **kwargs)


class Extractor(CCNN):
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)
        return x


class Classifier(CCNN):
    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return x


if __name__ == "__main__":
    dataset = DEAPDataset(
        io_path=f'./tmp_out/examples_quickstart_cross_subject/deap',
        root_path='./tmp_in/data_preprocessed_python',
        offline_transform=transforms.Compose([
            transforms.BandDifferentialEntropy(apply_to_baseline=True),
            transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT,
                              apply_to_baseline=True)
        ]),
        online_transform=transforms.Compose(
            [transforms.BaselineRemoval(),
             transforms.ToTensor()]),
        label_transform=transforms.Compose([
            transforms.Select('valence'),
            transforms.Binary(5.0),
        ]),
        num_worker=16)

    k_fold = LeaveOneSubjectOut(
        split_path=f'./tmp_out/examples_quickstart_cross_subject/split')

    scores = {}

    for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
        extractor = Extractor()
        classifier = Classifier()
        trainer = MyDDCTrainer(extractor=extractor,
                               classifier=classifier,
                               lr=1e-4,
                               weight_decay=1e-4,
                               device_ids=[0])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        trainer.fit(train_loader, val_loader, val_loader, num_epochs=50)
        scores[i] = trainer.score(val_loader)
