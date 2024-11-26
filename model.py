import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        image_channels = 1
        nb_channels = 32
        num_blocks = 6
        cond_channels = 16
        self.noise_emb = NoiseEmbedding(cond_channels)
        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(nb_channels, cond_channels) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)
    
    def forward(self, noisy_input: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        cond = self.noise_emb(c_noise)
        x = self.conv_in(noisy_input)
        for block in self.blocks:
            x = block(x, cond)
        return self.conv_out(x)


class NoiseEmbedding(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer('weight', torch.randn(1, cond_channels // 2))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 1
        f = 2 * torch.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


class ConditionalBatchNorm(nn.Module):
    def __init__(self, nb_channels, cond_channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(nb_channels, affine=False)
        self.mean = nn.Linear(cond_channels, nb_channels)
        self.var = nn.Linear(cond_channels, nb_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.norm(x) * self.var(cond).unsqueeze(2).unsqueeze(3) + self.mean(cond).unsqueeze(2).unsqueeze(3)

class ResidualBlock(nn.Module):
    def __init__(self, nb_channels: int, cond_channels) -> None:
        super().__init__()
        self.norm1 = ConditionalBatchNorm(nb_channels, cond_channels)
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = ConditionalBatchNorm(nb_channels, cond_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor, noise_emb: torch.Tensor) -> torch.Tensor:
        norm1 = self.norm1(x, noise_emb) 
        norm2 = self.norm2(x, noise_emb) 
        y = self.conv1(F.relu(norm1))
        y = self.conv2(F.relu(norm2))
        return x + y

