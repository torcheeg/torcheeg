import torch
from torchvision import transforms
from torcheeg.trainers import ADATrainer

from torchvision import datasets

import torch.utils.data as data
from PIL import Image
import os
import torch.nn as nn

class MINISTM(data.Dataset):

    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data


source_dataset = datasets.MNIST(
    root='./tmp_out/mnist',
    download=True,
    train=True,
    transform=transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))
    ]),
)
target_dataset = MINISTM(data_root='tmp_out/mnist_m/mnist_m_train',
                         data_list='tmp_out/mnist_m/mnist_m_train_labels.txt',
                         transform=transforms.Compose([
                             transforms.Resize(32),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                  std=(0.5, 0.5, 0.5))
                         ]))

val_dataset = MINISTM(data_root='tmp_out/mnist_m/mnist_m_test',
                      data_list='tmp_out/mnist_m/mnist_m_test_labels.txt',
                      transform=transforms.Compose([
                          transforms.Resize(32),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                               std=(0.5, 0.5, 0.5))
                      ]))

def conv2d(m,n,k,act=True):
    layers =  [nn.Conv2d(m,n,k,padding=1)]

    if act: layers += [nn.ELU()]

    return nn.Sequential(
        *layers
    )

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.features = nn.Sequential(
            nn.InstanceNorm2d(3),
            conv2d(3,  32, 3),
            conv2d(32, 32, 3),
            conv2d(32, 32, 3),
            nn.MaxPool2d(2, 2, padding=0),
            conv2d(32, 64, 3),
            conv2d(64, 64, 3),
            conv2d(64, 64, 3),
            nn.MaxPool2d(2, 2, padding=0),
            conv2d(64, 128, 3),
            conv2d(128, 128, 3),
            conv2d(128, 128, 3),
            nn.MaxPool2d(2, 2, padding=0)
        )
    
    def forward(self, x):
        x = x.expand(x.data.shape[0], 3, 32, 32)
        phi  = self.features(x)
        # phi_mean = phi.view(-1, 128, 16).mean(dim=-1)
        phi = phi.view(-1,128*4*4)
        return phi
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(128*4*4, 10)
        )
    
    def forward(self, phi):
        y = self.classifier(phi)
        return y

extractor = Extractor()
classifier = Classifier()

source_loader = data.DataLoader(source_dataset, batch_size=1000, shuffle=True, num_workers=4)
target_loader = data.DataLoader(target_dataset, batch_size=1000, shuffle=True, num_workers=4)
val_loader = data.DataLoader(val_dataset, batch_size=1000, shuffle=True, num_workers=4)
test_loader = data.DataLoader(val_dataset, batch_size=1000, shuffle=True, num_workers=4)

trainer = ADATrainer(extractor,
                     classifier,
                     num_classes=10,
                     devices=1,
                     visit_weight=0.6,
                     accelerator='gpu')
trainer.fit(source_loader, target_loader, target_loader, max_epochs=300)
trainer.test(target_loader)