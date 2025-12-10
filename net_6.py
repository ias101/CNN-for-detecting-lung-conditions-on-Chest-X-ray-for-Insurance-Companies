import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),  # Increased filter size
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1),  # Increased filter size
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1),  # Increased filter size
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Previous filter size
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Calculating the size of the feature maps after the last convolutional layer
        # For a 128x128 input image, the feature map size will be 8x8 after the last max pooling
        final_fm_size = 512 * 7 * 7
        
        self.linear_layers = nn.Sequential(
            nn.Linear(final_fm_size, 512),  # Adjust input size based on final feature map size
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
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x
