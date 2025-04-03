import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .DConv import SpatialConv
from .DTransformer import DTransformer
from .M2MTNetConfig import M2MTNetConfig

__all__ = ['M2MTNet']

SizeHW = Tuple[int, int]


class M2MTAttention(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, factor: int, patch_size: SizeHW, heads: int = 1,
                 sz_a: Tuple[int, int] = (5, 5)) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.sz_a = sz_a
        self.factor = factor
        in_channels_ = in_channels * sz_a[0] * sz_a[1]
        self.embed_dim = embed_dim

        self.norm = nn.LayerNorm(embed_dim)
        self.ff_in = nn.Linear(in_channels_, embed_dim, bias=True)
        nn.init.kaiming_uniform_(self.ff_in.weight, a=math.sqrt(5))
        self.qkv = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, bias=True),
            nn.Linear(embed_dim, embed_dim // factor**2, bias=True),
            nn.Linear(embed_dim, embed_dim // factor**2, bias=True)
        ])
        for w in self.qkv:
            nn.init.kaiming_uniform_(w.weight, a=math.sqrt(5))
        self.ff_out = nn.Linear(embed_dim, in_channels_, bias=True)
        nn.init.kaiming_uniform_(self.ff_out.weight, a=math.sqrt(5))

        self.heads = heads
        self.scale = (in_channels_ // heads) ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inp: torch.Tensor):
        B, C, N, H, W = inp.shape
        x = inp
        x = rearrange(x, 'b c n h w -> b (h w) (c n)')
        x = self.ff_in(x)
        x_norm = self.norm(x)
        q = self.qkv[0](x_norm)
        k = self.qkv[1](x_norm)
        v = self.qkv[2](x)
        q = rearrange(q, 'b hw1 (c1 head) -> b head hw1 c1', head=self.heads)
        k = rearrange(k, 'b (hw2 f) (c2 head) -> b head hw2 (c2 f)', head=self.heads, f=self.factor**2)
        v = rearrange(v, 'b (hw2 f) (c2 head) -> b head hw2 (c2 f)', head=self.heads, f=self.factor**2)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        x = attn @ v
        x = self.ff_out(x)
        x = rearrange(x, 'b head (h w) (c n) -> b (head c) n h w', n=N, h=H, w=W)
        return x


class M2MT(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=(32, 32), heads=1):
        super(M2MT, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attention = M2MTAttention(in_channels=in_channels, embed_dim=embed_dim, factor=1,
                                       patch_size=patch_size, heads=heads)
        self.sz_a = (5, 5)

    def forward(self, inp):
        buffer = rearrange(inp, 'b c u v h w -> b c (u v) h w', u=self.sz_a[0], v=self.sz_a[1])
        buffer = self.conv(buffer) + buffer

        buffer = self.attention(buffer) + buffer

        buffer = rearrange(buffer, 'b c (u v) h w -> b c u v h w', u=self.sz_a[0], v=self.sz_a[1])
        return buffer


class CorrBlock(nn.Module):
    def __init__(self, chns_in: int, embed_dim: int, patch_size: Tuple[int, int]) -> None:
        super(CorrBlock, self).__init__()
        self.body = [
            M2MT(in_channels=chns_in, embed_dim=embed_dim, patch_size=patch_size, heads=1),
            DTransformer(in_channels=chns_in, connection="uv")
        ]
        self.body = nn.Sequential(*self.body)

    def forward(self, x):
        x = self.body(x)
        return x


class M2MTNet(nn.Module):
    def __init__(self, scale: int, sz_a: tuple[int, int], config: M2MTNetConfig):
        super().__init__()
        self.scale = scale
        self.sz_a = sz_a
        self.config = config
        self.patch_size = config.patch_size

        self.n_chns_in = 1
        self.n_chns_ft = self.config.n_chns_ft
        self.embed_dim = self.config.embed_dim

        self.head = SpatialConv(in_channels=self.n_chns_in, out_channels=self.n_chns_ft, n=4, bias=False)
        self.body = []
        for _ in range(self.config.n_block):
            block = CorrBlock(
                chns_in=self.n_chns_ft, embed_dim=self.embed_dim, patch_size=self.patch_size
            )
            self.body.append(block)
        self.body = nn.ModuleList(self.body)

        self.tail = [
            nn.Conv2d(self.n_chns_ft, self.n_chns_ft * self.scale ** 2, kernel_size=1, padding=0, bias=False),
            nn.PixelShuffle(self.scale),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.n_chns_ft, self.n_chns_in, kernel_size=3, padding=1, bias=False),
        ]
        self.tail = nn.Sequential(*self.tail)

    def forward(self, inp, *args, **kwargs):
        u, v = self.sz_a
        inp = rearrange(inp, 'b c (u h) (v w) -> b c u v h w', u=u, v=v)

        x = self.head(inp)
        res = x
        for block in self.body:
            res = res + block(res)
        x = x + res

        x = rearrange(x, "b c u v h w -> b c (u h) (v w)")
        x = self.tail(x)

        sr = self.LF_interpolate(inp, self.scale, mode='bicubic')
        sr = rearrange(sr, "b c u v h w -> b c (u h) (v w)", u=u, v=v)
        x = sr + x
        return x

    @staticmethod
    def LF_interpolate(LF: torch.Tensor, scale: int, mode: str = 'bicubic') -> torch.Tensor:
        [b, c, u, v, h, w] = LF.size()
        LF = rearrange(LF, 'b c u v h w -> (b u v) c h w')
        LF_upscale = F.interpolate(LF, scale_factor=scale, mode=mode, align_corners=False)
        LF_upscale = rearrange(LF_upscale, '(b u v) c h w -> b c u v h w', u=u, v=v)
        return LF_upscale
