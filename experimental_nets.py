import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_1(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(Net_1, self).__init__()

        self.cnn_layers = nn.Sequential(            
            # Defining a 2D convolution layer
            nn.Conv2d(1, 64, kernel_size=4, stride=1, padding=1),  # Adjusting padding
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.5, inplace=True),
            
            # Defining a 2D convolution layer
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=1),  # Adjusting padding
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.5, inplace=True),
            
            # Defining another 2D convolution layer
            nn.Conv2d(64, 32, kernel_size=2, stride=1, padding=1),  # Adjusting padding
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.25, inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=1),  # Adjusting padding
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.25, inplace=True),
            
            # Defining another 2D convolution layer
            nn.Conv2d(32, 16, kernel_size=2, stride=1, padding=1),  # Adjusting padding
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.125, inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(16, 16, kernel_size=2, stride=1, padding=1),  # Adjusting padding
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.125, inplace=True),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(64, 256),
            nn.Linear(256, n_classes),
        )

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        # After our convolutional layers which are 2D, we need to flatten our
        # input to be 1 dimensional, as the linear layers require this.
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x



class Net_2(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(Net_2, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(256 * 15 * 15, 512), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, n_classes),
        )

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        # After our convolutional layers which are 2D, we need to flatten our
        # input to be 1 dimensional, as the linear layers require this.
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Net_3(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(Net_3, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            ResidualBlock(64, 128),
            nn.MaxPool2d(kernel_size=2),
            
            ResidualBlock(128, 256),
            nn.MaxPool2d(kernel_size=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(256 * 15 * 15, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, n_classes),
        )
            
    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        # After our convolutional layers which are 2D, we need to flatten our
        # input to be 1 dimensional, as the linear layers require this.
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    

class Net_5(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(Net_5, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        
        self.linear_layers = nn.Sequential(
            nn.Linear(256, 512),  # Adjust input size after global average pooling
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, n_classes),
        )

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x



class Net_7(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(Net_7, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=39, stride=1),  # Decreased filters from 90 to 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # Max pooling layer 1
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # Decreased filters from 180 to 128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # Max pooling layer 2
        )

        # Calculating the size of the feature maps after the last convolutional layer
        final_fm_size = 128 * 21 * 21  # Updated based on decreased filters
        
        self.linear_layers = nn.Sequential(
            nn.Linear(final_fm_size, 512),  # Fully connected layer 1
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),  # Fully connected layer 2
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, n_classes),  # Fully connected layer 3
        )

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x


class Net_8(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(Net_8, self).__init__()

        self.cnn_layers = nn.Sequential(
            
            # Defining a 2D convolution layer
            nn.Conv2d(1, 96, kernel_size=4, stride=1, padding=1),  # Adjusting padding
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.5, inplace=True),
            
            # Defining a 2D convolution layer
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),  # Adjusting padding
            # nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.375),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.3),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.225),
        )

        flattened_size = 16384

        self.linear_layers = nn.Sequential(
            nn.Linear(flattened_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, n_classes),
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
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class Net_9(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(Net_9, self).__init__()

        self.cnn_layers = nn.Sequential(

            # Defining a 2D convolution layer
            nn.Conv2d(1, 128, kernel_size=4, stride=1, padding=1),  # Adjusting padding
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.3, inplace=True),

            # Defining a 2D convolution layer
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),  # Adjusting padding
            # nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.5),

            nn.Conv2d(64, 96, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.42),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.125),
        )

        flattened_size = 16384

        self.linear_layers = nn.Sequential(
            nn.Linear(flattened_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, n_classes),
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
