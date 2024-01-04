import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Conv1dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, dilation=1, bias=True, **kwargs):
        super().__init__(**kwargs)
        assert dilation==1
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=groups, bias=bias)

    def calc_same_pad(self, i, k, s):
        # return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)
        if i%s == 0:
            pad = max(k - s, 0)
        else:
            pad = max(k - (i % s), 0)
        return pad

    def forward(self, inputs):
        x = inputs
        i = x.size()[-1]
        pad = self.calc_same_pad(i=i, k=self.kernel_size, s=self.stride)

        x = F.pad(x, [pad//2, pad - pad// 2])
        return self.conv(x)


class MaskedConv1d(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size=17,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv = Conv1dSame(
                        in_channels,
                        out_channels,
                        kernel_size,
                        groups=groups,
                        stride=stride,
                        dilation=dilation,
                        bias=bias)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
      if mask is not None:
        if self.stride > 1:
          mask = mask[:,::self.stride]
      return mask

    def forward(self, inputs, mask=None):
        x = inputs
        if mask is not None:
            x = x.masked_fill(~mask[:,None,:], torch.tensor(0., dtype=x.dtype, device=x.device))#torch.where(mask[:,None,:], x, torch.tensor(0., dtype=x.dtype, device=x.device))
        x = self.conv(x)
        return x

class Conv2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, dilation=1, bias=True, **kwargs):
        super().__init__(**kwargs)
        assert dilation==1
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=groups, bias=bias)

    def calc_same_pad(self, i, k, s):
        if i%s == 0:
            pad = max(k - s, 0)
        else:
            pad = max(k - (i % s), 0)
        return pad

    def forward(self, inputs):
        x = inputs
        h = x.size()[-2]
        w = x.size()[-1]
        padw = self.calc_same_pad(i=w, k=self.kernel_size, s=self.stride)
        padh = self.calc_same_pad(i=h, k=self.kernel_size, s=self.stride)
        x = F.pad(x, [padh//2, padh - padh// 2, padw//2, padw - padw // 2])
        return self.conv(x)

class MaskedConv2d(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv = Conv2dSame(
                        in_channels,
                        out_channels,
                        kernel_size,
                        groups=groups,
                        stride=stride,
                        dilation=dilation,
                        bias=bias)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
      if mask is not None:
        if self.stride > 1:
          mask = mask[:,::self.stride]
      return mask

    def forward(self, inputs, mask=None):
        x = inputs
        if mask is not None:
            x = x.masked_fill(~mask[:,None,None,:], torch.tensor(0., dtype=x.dtype, device=x.device))
            x = x.masked_fill(~mask[:,None,:,None], torch.tensor(0., dtype=x.dtype, device=x.device))
        x = self.conv(x)
        return x