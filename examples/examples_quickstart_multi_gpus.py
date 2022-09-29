import os

from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.models import CCNN
from torcheeg.trainers import ClassificationTrainer

if __name__ == "__main__":
    os.makedirs("./tmp_out/examples_quickstart_multi_gpus", exist_ok=True)

    dataset = DEAPDataset(
        io_path=f'./tmp_out/examples_quickstart_multi_gpus/deap',
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

    k_fold = KFoldGroupbyTrial(
        n_splits=5, split_path=f'./tmp_out/examples_quickstart_multi_gpus/split')

    scores = {}

    for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
        model = CCNN()
        trainer = ClassificationTrainer(model=model,
                                     lr=1e-4,
                                     weight_decay=1e-4,
                                     device_ids=[0, 1, 2])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        trainer.fit(train_loader, val_loader, num_epochs=50)
        scores[i] = trainer.score(val_loader)

# run the following shells to start, where nproc_per_node should be equal to the length of device_ids:

# python -m torch.distributed.launch \
#     --nproc_per_node=3 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr="localhost" \
#     --master_port=2345 \
#     examples/examples_quickstart_multi_gpus.py