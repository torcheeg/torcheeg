import torch
import torch.nn as nn
from .encoder import Encoder
from torch.autograd import Function
from .ae import AE

class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params
        self.ae = AE(params)
        self.classifier = nn.Linear(512, self.params.num_of_classes)

    def forward(self, x):
        recon, mu = self.ae(x)
        return self.classifier(mu), recon, mu

    def inference(self, x):
        mu = self.ae.encoder(x)
        return self.classifier(mu)


