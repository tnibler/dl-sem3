import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ConditionalBatchNorm, NoiseEmbedding

class UNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.name = 'UNet'
        img_c = 1
        noise_c = 32
        class_emb_dim = 8
        self.noise_emb = NoiseEmbedding(noise_c)
        self.class_emb = nn.Embedding(num_classes, class_emb_dim)

        unet_in_c = 16
        self.inconv = nn.Conv2d(img_c + class_emb_dim, unet_in_c, kernel_size=3, stride=1, padding=1)
        chans = [unet_in_c, unet_in_c * 2, unet_in_c * 4, unet_in_c * 8, unet_in_c * 8, unet_in_c * 16]
        self.down_blocks = nn.ModuleList([
            DownBlock(chans[i], chans[i+1], chans[i+1], noise_c)
            for i in range(len(chans) - 1)
        ])
        self.up_blocks = nn.ModuleList([
            UpBlock(
                chans[i] if i == len(chans) - 1 else 2 * chans[i],
                chans[i], chans[i - 1], noise_c)
            for i in reversed(range(1, len(chans)))
        ])
        self.outconv = nn.Conv2d(2*chans[0], img_c, kernel_size=3, stride=1, padding=1)
    
    def forward(self, input: torch.Tensor, c: torch.Tensor, lbl: torch.Tensor):
        assert len(input.shape) == 4, f"shape must be BxCxHxW, got {input.shape}"
        bs, _, h, w = input.shape
        c = self.noise_emb(c)
        class_emb = self.class_emb(lbl)
        class_emb = class_emb.reshape(bs, class_emb.shape[1], 1, 1).expand(bs, class_emb.shape[1], h, w)
        y = self.inconv(torch.cat([input, class_emb], dim=1))
        res = y.clone()
        down_out = []
        for block in self.down_blocks[:-1]:
            y = block(y, c)
            down_out.append(y.clone())
        y = self.down_blocks[-1](y, c)
        y = self.up_blocks[0](y, c)
        for block, skip in zip(self.up_blocks[1:], reversed(down_out)):
            y = block(torch.cat([y, skip], dim=1), c)
        y = self.outconv(torch.cat([y, res], dim=1))
        return y

class DownBlock(nn.Module):
    """
    Sequence of MaxPool -> (Conv2D -> BatchNorm) -> (Conv2D -> BatchNorm) with ReLU

    in_c: number of input channels
    mid_c: number of output channels for first conv
    out_c: number of output channels
    cond_c: number of channels of noise conditioning
    """
    def __init__(self, in_c, inter_c, out_c, cond_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, inter_c, kernel_size=3, padding=1)
        self.bn1 = ConditionalBatchNorm(inter_c, cond_c)
        self.conv2 = nn.Conv2d(inter_c, out_c, kernel_size=3, padding=1)
        self.bn2 = ConditionalBatchNorm(out_c, cond_c)
        self.pool = nn.MaxPool2d(2)
        # if in_c == out_c:
        #     self.res_adapt = nn.Identity()
        # else:
        #     self.res_adapt = nn.Conv2d(in_c, out_c, kernel_size=1)
    
    def forward(self, x, c):
        # res = self.res_adapt(y)
        y = self.conv1(x)
        y = self.bn1(F.relu(y), c)
        y = self.conv2(y)
        y = self.bn2(F.relu(y), c)
        y = self.pool(y)
        return y 

class UpBlock(nn.Module):
    """
    Sequence of ConvTranspose -> (Conv2D -> BatchNorm) -> (Conv2D -> BatchNorm)

    in_c: number of input channels
    mid_c: number of output channels for first conv
    out_c: number of output channels
    cond_c: number of channels of noise conditioning
    """
    def __init__(self, in_c, inter_c, out_c, cond_c):
        super().__init__()
        self.convt = nn.ConvTranspose2d(in_c, in_c, kernel_size=2, stride=2)
        # self.convt = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_c, inter_c, kernel_size=3, padding=1)
        self.bn1 = ConditionalBatchNorm(inter_c, cond_c)
        self.conv2 = nn.Conv2d(inter_c, out_c, kernel_size=3, padding=1)
        self.bn2 = ConditionalBatchNorm(out_c, cond_c)
        if in_c == out_c:
            self.res_adapt = nn.Identity()
        else:
            self.res_adapt = nn.Conv2d(in_c, out_c, kernel_size=1)
    
    def forward(self, x, c):
        y = self.convt(x)
        res = self.res_adapt(y)
        y = self.conv1(y)
        y = self.bn1(F.relu(y), c)
        y = self.conv2(y)
        y = self.bn2(F.relu(y), c)
        return y + res