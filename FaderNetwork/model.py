import torch
from torch import nn


def Conv_BN_ReLU(in_channels, num_of_filters):
    return nn.Sequential(
        nn.Conv2d(in_channels, num_of_filters, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(num_features=num_of_filters),
        nn.LeakyReLU(0.2)
    )


def deConv_BN_ReLU(in_channels, num_of_filters):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, num_of_filters, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(num_features=num_of_filters),
        nn.ReLU()
    )


class ConvAutoEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super(ConvAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            Conv_BN_ReLU(3, 16),
            Conv_BN_ReLU(16, 32),
            Conv_BN_ReLU(32, 64),
            Conv_BN_ReLU(64, 128),
            Conv_BN_ReLU(128, 256),
            Conv_BN_ReLU(256, 512),
            Conv_BN_ReLU(512, 512),
        )
        self.decoder = nn.Sequential(
            deConv_BN_ReLU(512 + 1, 512 + 1),
            deConv_BN_ReLU(512 + 1, 256 + 1),
            deConv_BN_ReLU(256 + 1, 128 + 1),
            deConv_BN_ReLU(128 + 1, 64 + 1),
            deConv_BN_ReLU(64 + 1, 32 + 1),
            deConv_BN_ReLU(32 + 1, 16 + 1),
            deConv_BN_ReLU(16 + 1, 3)
        )
        if pretrained:
            model = torch.load('./5k_images_gender_model.pth', map_location='cpu')
            self.encoder = model.encoder
            self.decoder = model.decoder

    def forward(self, img, features):
        return self.decoder(torch.cat((self.encoder(img), features), dim=1))
