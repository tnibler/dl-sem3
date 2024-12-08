import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ConditionalBatchNorm, NoiseEmbedding

class UNet(nn.Module):
    def __init__(self, num_classes: int|None):
        super().__init__()
        self.name = 'UNet'
        img_c = 1
        noise_c = 32
        self.noise_emb = NoiseEmbedding(noise_c)
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes, class_emb_dim)
            class_emb_dim = 16
        else:
            class_emb_dim = 0
            self.class_emb = None

        unet_in_c = 64
        self.inconv = nn.Conv2d(img_c + class_emb_dim, unet_in_c, kernel_size=3, stride=1, padding=1)
        chans = [unet_in_c, unet_in_c * 4, unet_in_c * 8, unet_in_c * 8, unet_in_c * 8, unet_in_c * 16]
        self.down_blocks = nn.ModuleList([
            DownBlock(chans[i], chans[i+1], chans[i+1], noise_c + class_emb_dim)
            for i in range(len(chans) - 1)
        ])
        self.conv_between1 = nn.Conv2d(chans[-1], 1024, kernel_size=3, padding=1)
        self.bn_between1 = ConditionalBatchNorm(1024, noise_c + class_emb_dim)
        self.conv_between2 = nn.Conv2d(1024, chans[-1], kernel_size=3, padding=1)
        self.bn_between2 = ConditionalBatchNorm(chans[-1], noise_c + class_emb_dim)
        self.up_blocks = nn.ModuleList([
            UpBlock(
                chans[i] if i == len(chans) - 1 else 2 * chans[i],
                chans[i], chans[i - 1], noise_c + class_emb_dim)
            for i in reversed(range(1, len(chans)))
        ])
        self.outconv = nn.Conv2d(2*chans[0], img_c, kernel_size=3, stride=1, padding=1)
    
    def forward(self, input: torch.Tensor, noise_cond: torch.Tensor, lbl: torch.Tensor | None):
        assert len(input.shape) == 4, f"shape must be BxCxHxW, got {input.shape}"
        bs, _, h, w = input.shape
        noise_cond = self.noise_emb(noise_cond)
        if self.class_emb and lbl is not None:
            class_emb = self.class_emb(lbl)
            conds = torch.cat([noise_cond, class_emb], dim=1)
            class_emb = class_emb.reshape(bs, class_emb.shape[1], 1, 1).expand(bs, class_emb.shape[1], h, w)
        else:
            conds = noise_cond

        y = self.inconv(torch.cat([input, class_emb], dim=1) if self.class_emb else input)
        res = y.clone()
        down_out = []
        for block in self.down_blocks[:-1]:
            y = block(y, conds)
            down_out.append(y.clone())
        y = self.down_blocks[-1](y, conds)
        y = self.conv_between1(y)
        y = self.bn_between1(F.relu(y), conds)
        y = self.conv_between2(y)
        y = self.bn_between2(F.relu(y), conds)
        y = self.up_blocks[0](y, conds)
        for block, skip in zip(self.up_blocks[1:], reversed(down_out)):
            y = block(torch.cat([y, skip], dim=1), conds)
        y = self.outconv(torch.cat([y, res], dim=1))
        return y

class ResBlock(nn.Module):
    """
    Sequence of (Conv2D -> BatchNorm) -> (Conv2D -> BatchNorm) with ReLU and residula connection

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
        if in_c == out_c:
            self.res_adapt = nn.Identity()
        else:
            self.res_adapt = nn.Conv2d(in_c, out_c, kernel_size=1)
    
    def forward(self, x, c):
        res = self.res_adapt(x)
        y = self.conv1(x)
        y = self.bn1(F.relu(y), c)
        y = self.conv2(y)
        y = self.bn2(F.relu(y), c)
        return y + res

class DownBlock(nn.Module):
    def __init__(self, in_c, inter_c, out_c, cond_c):
        super().__init__()
        self.res_block = ResBlock(in_c, inter_c, inter_c, cond_c)
        self.down = nn.MaxPool2d(2)
        self.outconv = nn.Conv2d(inter_c, out_c, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, c):
        y = self.res_block(x, c)
        y = self.down(y)
        y = self.outconv(y)
        return y

class UpBlock(nn.Module):
    def __init__(self, in_c, inter_c, out_c, cond_c):
        super().__init__()
        self.inconv = nn.Conv2d(in_c, inter_c, kernel_size=3, stride=1, padding=1)
        self.convt = nn.ConvTranspose2d(inter_c, inter_c, kernel_size=2, stride=2)
        self.res_block = ResBlock(inter_c, inter_c, out_c, cond_c)
    
    def forward(self, x, c):
        y = self.inconv(x)
        y = self.convt(y)
        y = self.res_block(y, c)
        return y