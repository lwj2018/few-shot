import torch.nn as nn
import torch
import torch.nn.functional as F

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def conv_block_1d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 1, padding=0),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )


class Convnet(nn.Module):

    def __init__(self, out_dim, x_dim=3, hid_dim=64, z_dim=64, feature_dim = 1600):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.feature_dim = feature_dim
        self.fc = nn.Linear(feature_dim,out_dim)

    def get_feature(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.get_feature(x)
        x = self.fc(x)
        return x