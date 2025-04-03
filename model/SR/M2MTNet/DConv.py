from typing import Callable, Tuple, TypeVar
import torch
import torch.nn as nn
from einops import rearrange

__all__ = ['DConv', 'SpatialConv', 'AngularConv']


T = TypeVar('T')
_spatial = Tuple[T, T]


class DConv(nn.Module):
    available_dims = ['u', 'v', 'h', 'w']
    all_dims = ['b', 'c'] + available_dims

    def __init__(self,
                 in_channels: int, out_channels: int, connection: str,
                 kernel_size: _spatial[int] = (3, 3),
                 stride: _spatial[int] = (1, 1),
                 padding: _spatial[int] = (1, 1),
                 n: int = 1,
                 act: Callable[[], nn.Module] = lambda: nn.LeakyReLU(0.2, inplace=True),
                 *args, **kwargs):
        super(DConv, self).__init__()
        assert all(dim in self.available_dims for dim in connection)
        self.connection = connection
        self.dims_preserve, self.dims_active = [], []
        for dim in self.available_dims:
            _dst = self.dims_active if dim in connection else self.dims_preserve
            _dst.append(dim)
        self.pattern0 = f"{' '.join(['b', 'c'] + self.available_dims)}"
        self.pattern1 = f"({' '.join(['b'] + self.dims_preserve)}) {' '.join(['c'] + self.dims_active)}"

        self.body = []
        for i in range(n):
            self.body.append(
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          *args, **kwargs))
            if act:
                self.body.append(act())
        self.body = nn.Sequential(*self.body)

    def forward(self, inp: torch.Tensor):
        szs_preserve = {dim: sz for dim, sz in zip(self.all_dims, inp.shape) if dim in self.dims_preserve}
        x = rearrange(inp, f'{self.pattern0} -> {self.pattern1}')
        x = self.body(x)
        x = rearrange(x, f'{self.pattern1} -> {self.pattern0}', **szs_preserve)

        return x


def SpatialConv(*args, **kwargs) -> DConv:
    return DConv(connection='hw', *args, **kwargs)


def AngularConv(*args, **kwargs) -> DConv:
    return DConv(connection='uv', *args, **kwargs)
