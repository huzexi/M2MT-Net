import math
import torch
import torch.nn as nn
from einops import rearrange

from .DConv import SpatialConv


__all__ = ['DTransformer']


def lf2token(lf: torch.Tensor) -> torch.Tensor:
    token = rearrange(lf, 'b c d d3 d4 -> d (b d3 d4) c')
    return token


def token2lf(token: torch.Tensor, d1: int, d2: int, d3: int, d4: int) -> torch.Tensor:
    lf = rearrange(token, '(d3 d4) (b d1 d2) (c) -> b c d1 d2 d3 d4', d1=d1, d2=d2, d3=d3, d4=d4)
    return lf


class DTransformer(nn.Module):
    available_dims = ['u', 'v', 'h', 'w']
    all_dims = ['b', 'c'] + available_dims

    def __init__(self, in_channels: int, connection: str, kernel_sz: int = 3):
        super(DTransformer, self).__init__()
        assert all(dim in self.available_dims for dim in connection)
        self.connection = connection
        self.kernel_sz = kernel_sz
        self.dims_preserve, self.dims_active = [], []
        for dim in self.available_dims:
            _dst = self.dims_active if dim in connection else self.dims_preserve
            _dst.append(dim)
        self.pattern0 = f"{' '.join(['b', 'c'] + self.available_dims)}"
        self.pattern1 = f"{' '.join(['b', 'c'] + self.dims_preserve + self.dims_active)}"

        # if "w" in self.dims_active or "h" in self.dims_active:
        self.linear_in = nn.Linear(in_channels, in_channels, bias=True)
        self.norm = nn.LayerNorm(in_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=8,
            bias=True
        )
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * 2, bias=True),
            nn.ReLU(True),
            nn.Linear(in_channels * 2, in_channels, bias=True),
        )
        self.linear_out = nn.Linear(in_channels, in_channels, bias=True)
        self.conv = SpatialConv(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                n=3, bias=True)

    def forward(self, inp: torch.Tensor):
        szs_preserve = {dim: sz for dim, sz in zip(self.all_dims, inp.shape) if dim in self.dims_preserve}
        x = rearrange(inp, f'{self.pattern0} -> {self.pattern1}')
        _, c, d1, d2, d3, d4 = x.shape

        x = rearrange(x, 'b c d1 d2 d3 d4 -> b c (d3 d4) d1 d2')
        token = lf2token(x)
        token = self.linear_in(token)

        token_norm = self.norm(token)
        token = self.attention(query=token_norm,
                               key=token_norm,
                               value=token,
                               need_weights=False)[0] + token
        token = self.feed_forward(token) + token
        token = self.linear_out(token)

        x = token2lf(token, d1=d1, d2=d2, d3=d3, d4=d4)
        x = rearrange(x, f'{self.pattern1} -> {self.pattern0}', **szs_preserve)
        x = self.conv(x) + x
        return x
