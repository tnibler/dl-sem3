import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplestModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        image_channels = 1
        nb_channels = 16
        num_blocks = 5
        self.name = "SimplestModel"
        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(nb_channels) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)
    
    def forward(self, noisy_input: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        cond = self.noise_emb(c_noise)
        x = self.conv_in(noisy_input)
        for block in self.blocks:
            x = block(x, cond)
        return self.conv_out(x)

class ResidualBlockNoCond(nn.Module):
    def __init__(self, nb_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(nb_channels)
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor, noise_emb: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x, noise_emb) 
        y = self.conv1(F.relu(y))
        y = self.norm2(y, noise_emb)
        y = self.conv2(F.relu(y))
        return x + y


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
        var = F.relu(self.var(cond))
        mean = F.relu(self.mean(cond))
        return self.norm(x) * var.unsqueeze(2).unsqueeze(3) + mean.unsqueeze(2).unsqueeze(3)

class ResidualBlock(nn.Module):
    def __init__(self, nb_channels: int, cond_channels) -> None:
        super().__init__()
        self.norm1 = ConditionalBatchNorm(nb_channels, cond_channels)
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = ConditionalBatchNorm(nb_channels, cond_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor, noise_emb: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x, noise_emb) 
        y = self.conv1(F.relu(y))
        y = self.norm2(y, noise_emb)
        y = self.conv2(F.relu(y))
        return x + y

class SimpleResnet(nn.Module):
    def __init__(self, chans, spatial_encoding=False, device=None) -> None:
        super().__init__()
        image_channels = 1
        in_channels = image_channels
        if spatial_encoding:
            grid_y, grid_x = torch.meshgrid(torch.arange(0, 32), torch.arange(0, 32), indexing='ij')
            grid_y = (grid_y / 32) - .5
            grid_x = (grid_x / 32) - .5
            angle = torch.atan2(grid_y, grid_x)
            mag = (grid_y.pow(2) + grid_x.pow(2)).sqrt()
            self.spatial_enc = torch.stack([grid_x, grid_y, angle, mag]).reshape(1, 4, 32, 32).to(device)
            in_channels = image_channels + 4
        else:
            self.spatial_enc = None

        cond_channels = 16
        self.noise_emb = NoiseEmbedding(cond_channels)
        self.name = f'SimpleUnetNoDownsample{"-".join(map(str, chans))}'
        self.in_conv = nn.Conv2d(in_channels, chans[0], kernel_size=3, stride=1, padding=1)
        self.down_blocks = nn.ModuleList([
            ResBlock(chans[i], chans[i+1], cond_channels)
            for i in range(len(chans) - 1)
        ])
        self.up_blocks = nn.ModuleList([
            ResBlock(chans[i]*2, chans[i-1], cond_channels)
            for i in reversed(range(1, len(chans)))
        ])
        self.out_conv = nn.Conv2d(chans[0], image_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        if self.spatial_enc is not None:
            x = torch.cat([x, self.spatial_enc.expand((x.shape[0], 4, 32, 32)).to(x)], dim=1)
        cond = self.noise_emb(c_noise)
        x = self.in_conv(x)
        down_results = []
        for block in self.down_blocks:
            x = block(x, cond)
            down_results.append(x)
        for i, block in enumerate(self.up_blocks):
            x = block(torch.cat([x, down_results[-i-1]], dim=1), cond)
        return self.out_conv(x) 

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, cond_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)
        self.norm1 = ConditionalBatchNorm(out_c, cond_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.norm2 = ConditionalBatchNorm(out_c, cond_c)
        self.res_adapt = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1)
    
    def forward(self, x, cond):
        y = self.conv1(x)
        y = self.norm1(F.relu(y), cond)
        y = self.conv2(y)
        y = self.norm2(F.relu(y), cond)
        return y + self.res_adapt(x)

class SimpleResnetClassCond(nn.Module):
    def __init__(self, chans, num_classes, spatial_encoding=False, device=None) -> None:
        super().__init__()
        image_channels = 1
        in_channels = image_channels
        if spatial_encoding:
            grid_y, grid_x = torch.meshgrid(torch.arange(0, 32), torch.arange(0, 32), indexing='ij')
            grid_y = (grid_y / 32) - .5
            grid_x = (grid_x / 32) - .5
            angle = torch.atan2(grid_y, grid_x)
            mag = (grid_y.pow(2) + grid_x.pow(2)).sqrt()
            self.spatial_enc = torch.stack([grid_x, grid_y, angle, mag]).reshape(1, 4, 32, 32).to(device)
            in_channels = image_channels + 4
        else:
            self.spatial_enc = None

        self.class_emb = nn.Embedding(num_classes, 4)
        cond_channels = 16
        self.noise_emb = NoiseEmbedding(cond_channels)
        self.name = f'SimpleResnetClassCond{"-spat-" if spatial_encoding else ""}{"-".join(map(str, chans))}'
        self.in_conv = nn.Conv2d(in_channels + self.class_emb.embedding_dim, chans[0], kernel_size=3, stride=1, padding=1)
        self.down_blocks = nn.ModuleList([
            ResBlock(chans[i], chans[i+1], cond_channels)
            for i in range(len(chans) - 1)
        ])
        self.up_blocks = nn.ModuleList([
            ResBlock(chans[i]*2, chans[i-1], cond_channels)
            for i in reversed(range(1, len(chans)))
        ])
        self.out_conv = nn.Conv2d(chans[0], image_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor, c_noise: torch.Tensor, class_lbls) -> torch.Tensor:
        bs = x.shape[0]
        if self.spatial_enc is not None:
            x = torch.cat([x, self.spatial_enc.expand((bs, 4, 32, 32))], dim=1)
        class_emb = self.class_emb(class_lbls)
        class_emb = class_emb.reshape(bs, class_emb.shape[1], 1, 1).expand(bs, class_emb.shape[1], 32, 32)
        cond = self.noise_emb(c_noise)
        x = self.in_conv(torch.cat([x, class_emb], dim=1))
        down_results = []
        for block in self.down_blocks:
            x = block(x, cond)
            down_results.append(x)
        for i, block in enumerate(self.up_blocks):
            x = block(torch.cat([x, down_results[-i-1]], dim=1), cond)
        return self.out_conv(x) 