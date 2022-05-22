import torch
import torch.nn as nn


class Generator1(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, *512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.layer1 = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=2048, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=2048),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True))
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True))
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True))
        self.final = nn.Sequential(
            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=128, out_channels=channels, kernel_size=4, stride=2, padding=1))
        # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.final(x)
        return self.output(x)


class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, *512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.layer1 = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True))
        self.layer4= nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True))
        self.final = nn.Sequential(
            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=64, out_channels=channels, kernel_size=4, stride=2, padding=1))

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.layer5(x)
        x = self.final(x)
        return self.output(x)