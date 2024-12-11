import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplestModel(nn.Module):
    """
    ResNet as given in the instructions.
    """
    def __init__(self) -> None:
        super().__init__()
        image_channels = 1
        nb_channels = 16
        num_blocks = 5
        self.name = "SimplestModel"
        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlockNoCond(nb_channels) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)
    
    def forward(self, noisy_input: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(noisy_input)
        for block in self.blocks:
            x = block(x)
        return self.conv_out(x)

class ResidualBlockNoCond(nn.Module):
    def __init__(self, nb_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(nb_channels)
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x) 
        y = self.conv1(F.relu(y))
        y = self.norm2(y)
        y = self.conv2(F.relu(y))
        return x + y
