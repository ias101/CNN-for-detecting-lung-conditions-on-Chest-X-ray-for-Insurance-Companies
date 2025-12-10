from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, stride=1, id=0):
        super(ResidualBlock, self).__init__()
        self.id = id
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[1], stride=stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_sizes[1], stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[1], stride=stride, padding=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Net(nn.Module):

    def __init__(
            self,
            n_classes: int,
            in_channels: int = 1,
            base_channels: int = 64,
            residual_blocks: int = 1,
            pooling_type: str = 'avg',
            pooling_kernel_size: List[int] = None,
            dropout_p1: float = 0.5,
            dropout_p2: float = 0.25,
            kernel_sizes: List[int] = None,
            num_layers: int = 1
    ) -> None:
        super(Net, self).__init__()
        self.num_layers = num_layers
        if kernel_sizes is None:
            kernel_sizes = [3] * (residual_blocks * 2 + 1)
        if pooling_kernel_size is None:
            pooling_kernel_size = [2] * (residual_blocks + 1)

        if pooling_type == 'max':
            pool_layer = nn.MaxPool2d
        elif pooling_type == 'avg':
            pool_layer = nn.AvgPool2d
        else:
            raise ValueError("Invalid pooling type. Use 'max' or 'avg'.")

        #  init layer
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=kernel_sizes[0], stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            pool_layer(kernel_size=pooling_kernel_size[0])
        )

        for i in range(num_layers):
            if i == 0:
                in_channels = base_channels
                out_channels = base_channels
            else:
                in_channels = base_channels
                out_channels = base_channels * 2
                base_channels *= 2

            # Add the first residual block for this layer
            self.cnn_layers.add_module(f'ResidualBlock_{i}',
                                       ResidualBlock(in_channels, out_channels, kernel_sizes=kernel_sizes))

            # Add the remaining residual blocks for this layer
            for j in range(residual_blocks - 1):
                self.cnn_layers.add_module(f'ResidualBlock_{j}_{i}',
                                          ResidualBlock(out_channels, out_channels, kernel_sizes=kernel_sizes, id=j))

            # Add the pooling layer for this layer
            self.cnn_layers.add_module(
                f'{pooling_type.capitalize()}Pool2d_{i}',
                pool_layer(kernel_size=pooling_kernel_size[1], stride=pooling_kernel_size[1], padding=0,
                           ceil_mode=False)  # 1 + i
            )

        def cal():
            x = 128 - kernel_sizes[0] + 2 + 1
            x = x // pooling_kernel_size[0]
            for _ in range(num_layers):
                x = x - kernel_sizes[1] + 2 + 1
                x = x - kernel_sizes[1] + 2 + 1
                x = x // pooling_kernel_size[1]
            return x

        self.linear_layers = nn.Sequential(
            nn.Linear(base_channels * (cal())** 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
