import torch
import torch.nn as nn
import torch.nn.functional as F


class AGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_t = nn.Conv2d(in_channels, out_channels, kernel_size=(9, 1), padding=(4, 0))
        self.conv_v = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv_t(x)
        x = self.conv_v(x)
        x = self.bn(x)
        return F.relu(x)


class AGCNModel(nn.Module):

    def __init__(self, in_channels=3, hidden=64, out_dim=512):
        super().__init__()

        self.data_bn = nn.BatchNorm1d(in_channels * 25)

        self.block1 = AGCNBlock(in_channels, hidden)
        self.block2 = AGCNBlock(hidden, hidden * 2)
        self.block3 = AGCNBlock(hidden * 2, hidden * 4)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden * 4, out_dim)

    def forward(self, x):
        B, T, V, C = x.shape

        x = x.permute(0, 3, 1, 2).contiguous()

        x = x.view(B, C * V, T)
        x = self.data_bn(x)
        x = x.view(B, C, V, T).permute(0, 1, 3, 2)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.pool(x).view(B, -1)
        x = self.fc(x)
        return x
