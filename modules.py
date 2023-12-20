from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
import math


def get_act_fn(activation):
    if activation == 'swish':
        return nn.SiLU()
    elif activation == 'silu':
        return nn.SiLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'mish':
        return nn.Mish()
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'elu':
        return nn.ELU()
    else:
        raise NotImplmentedError
        
# Masked Batch Normalization


def masked_batch_norm(input: Tensor, mask: Tensor, weight: Optional[Tensor], bias: Optional[Tensor],
                      running_mean: Optional[Tensor], running_var: Optional[Tensor], training: bool,
                      momentum: float, eps: float = 1e-5) -> Tensor:
    #from https://gist.github.com/yangkky/364413426ec798589463a3a88be24219
    r"""Applies Masked Batch Normalization for each channel in each data sample in a batch.
    See :class:`~MaskedBatchNorm1d`, :class:`~MaskedBatchNorm2d`, :class:`~MaskedBatchNorm3d` for details.
    """
    if not training and (running_mean is None or running_var is None):
        raise ValueError('Expected running_mean and running_var to be not None when training=False')

    num_dims = len(input.shape[2:])
    _dims = (0,) + tuple(range(-num_dims, 0))
    _slice = (None, ...) + (None,) * num_dims

    if training:
        num_elements = mask.sum(_dims)
        mean = (input * mask).sum(_dims) / num_elements  # (C,)
        var = (((input - mean[_slice]) * mask) ** 2).sum(_dims) / num_elements  # (C,)

        if running_mean is not None:
            running_mean.copy_(running_mean * (1 - momentum) + momentum * mean.detach())
        if running_var is not None:
            running_var.copy_(running_var * (1 - momentum) + momentum * var.detach())
    else:
        mean, var = running_mean, running_var

    out = (input - mean[_slice]) / torch.sqrt(var[_slice] + eps)  # (N, C, ...)

    if weight is not None and bias is not None:
        out = out * weight[_slice] + bias[_slice]

    return out


class _MaskedBatchNorm(_BatchNorm):
    #from https://gist.github.com/yangkky/364413426ec798589463a3a88be24219
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_MaskedBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input: Tensor, mask: Tensor = None) -> Tensor:
        self._check_input_dim(input)
        if mask is not None:
            self._check_input_dim(mask)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if mask is None:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps
            )
        else:
            return masked_batch_norm(
                input, mask, self.weight, self.bias,
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                bn_training, exponential_average_factor, self.eps
            )


class MaskedBatchNorm1d(torch.nn.BatchNorm1d, _MaskedBatchNorm):
    #from https://gist.github.com/yangkky/364413426ec798589463a3a88be24219
    r"""Applies Batch Normalization over a masked 3D input
    (a mini-batch of 1D inputs with additional channel dimension)..
    See documentation of :class:`~torch.nn.BatchNorm1d` for details.
    Shape:
        - Input: :math:`(N, C, L)`
        - Mask: :math:`(N, 1, L)`
        - Output: :math:`(N, C, L)` (same shape as input)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True, channels_last: bool = False) -> None:
        super(MaskedBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.channels_last = channels_last

    def forward(self, inputs, mask=None):
        if self.channels_last:
            inputs = inputs.permute(0,2,1)
        if mask is not None:
            mask = mask[:,None,:]
        out = super(MaskedBatchNorm1d, self).forward(inputs, mask)
        if self.channels_last:
            out = out.permute(0,2,1)
        return out


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
        
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x, mask=None):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class ScaleBiasLayer(nn.Module):
    """
    Computes an affine transformation y = x * scale + bias, either learned via adaptive weights, or fixed.
    Efficient alternative to LayerNorm where we can avoid computing the mean and variance of the input, and
    just rescale the output of the previous layer.

    Args:
        d_model (int): input dimension of layer.
        adaptive_scale (bool): whether to learn the affine transformation parameters or not. If set to False,
            the scale is fixed to 1 and bias to 0, effectively performing a No-Op on the input.
            This is done for export compatibility.
    """

    def __init__(self, d_model: int, adaptive_scale: bool):
        super().__init__()
        self.adaptive_scale = adaptive_scale
        if adaptive_scale:
            self.scale = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_buffer('scale', torch.ones(d_model), persistent=True)
            self.register_buffer('bias', torch.zeros(d_model), persistent=True)

    def forward(self, x):
        scale = self.scale.view(1, 1, -1)
        bias = self.bias.view(1, 1, -1)
        return x * scale + bias


class GLU(nn.Module):
    def __init__(self, dim: int, activation: str = 'sigmoid') -> None:
        super(GLU, self).__init__()
        self.dim = dim
        self.activation = get_act_fn(activation)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * self.activation(gate)

class Mlp(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        expand: int = 4,
        dropout: float = 0.1,
        bias : bool = True,
        activation: str = 'gelu'
    ) -> None:
        super(Mlp, self).__init__()

        self.ffn1 = nn.Linear(dim, dim * expand, bias=bias)
        self.act = get_act_fn(activation)
        self.do1 = nn.Dropout(p=dropout)
        self.ffn2 = nn.Linear(dim * expand, dim, bias=bias)
        # self.do2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        # x = self.do2(x)

        return x

class GLUMlp(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        expand: int = 4,
        dropout: float = 0.1,
        bias : bool = True,
        activation: str = 'gelu'
    ) -> None:
        super(GLUMlp, self).__init__()

        self.ffn1 = nn.Linear(dim, dim * expand, bias=bias)
        self.glu = GLU(dim=-1, activation=activation)
        self.do1 = nn.Dropout(p=dropout)
        self.ffn2 = nn.Linear(dim * expand // 2, dim, bias=bias)
        # self.do2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.glu(x)
        x = self.do1(x)
        x = self.ffn2(x)
        # x = self.do2(x)

        return x


class MaskedSoftmax(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        # self.softmax = nn.Softmax(self.dim)

    def forward(self, inputs, mask=None):
        if mask is not None:
            # Since mask is 1.0 for positions we want to keep and 0.0 for masked
            # positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -1e.9 for masked positions.
            # adder = (1.0 - mask.to(inputs.dtype)) * (
            #     torch.finfo(inputs.dtype).min
            # )

            # Since we are adding it to the raw scores before the softmax, this
            # is effectively the same as removing these entirely.
            # inputs += adder
            inputs = inputs.masked_fill(~mask, torch.finfo(inputs.dtype).min)
        return F.softmax(inputs, dim=self.dim)#, dtype=torch.float32)


def get_alibi(
    max_positions: int,
    attention_heads: int,
    dims: int = 1,
    distance: str = "manhattan",
):
    #from https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/utils.py
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        # In the paper, we only train models that have 2^a heads for some
        # a. This function has some good properties that only occur when
        # the input is a power of 2. To maintain that even when the number
        # of heads is not a power of 2, we use this workaround.
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    maxpos = max_positions
    attn_heads = attention_heads
    slopes = torch.Tensor(get_slopes(attn_heads))

    if dims == 1:
        # prepare alibi position linear bias. Note that wav2vec2 is non
        # autoregressive model so we want a symmetric mask with 0 on the
        # diagonal and other wise linear decreasing valuees
        pos_bias = (
            torch.abs(
                torch.arange(maxpos).unsqueeze(0) - torch.arange(maxpos).unsqueeze(1)
            )
            * -1
        )
    elif dims == 2:
        if distance == "manhattan":
            df = lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2)
        elif distance == "euclidean":
            df = lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        n = math.sqrt(max_positions)
        assert n.is_integer(), n
        n = int(n)

        pos_bias = torch.zeros((max_positions, max_positions))

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        new_x = i * n + j
                        new_y = k * n + l
                        pos_bias[new_x, new_y] = -df(i, j, k, l)

    else:
        raise Exception(f"unsupported number of alibi dims: {dims}")

    alibi_bias = slopes.unsqueeze(1).unsqueeze(1) * pos_bias.unsqueeze(0).expand(
        attn_heads, -1, -1
    )

    return alibi_bias


def get_alibi_bias(
    alibi_biases,
    batch_size,
    time_steps,
    heads,
    dtype,
    device,
    dims=1,
    distance="manhattan",
):
    #from https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/utils.py
    cache_key = f"{dims}_{heads}_{distance}"

    buffered = alibi_biases.get(cache_key, None)

    target_size = heads * batch_size
    if (
        buffered is None
        or buffered.size(0) < target_size
        or buffered.size(1) < time_steps
        or buffered.dtype != dtype
        or buffered.device != device
    ):
        bt = max(time_steps, buffered.size(1) if buffered is not None else 0)
        bn = max(target_size, buffered.size(0) if buffered is not None else 0) // heads

        buffered = (
            get_alibi(bt, heads, dims=dims, distance=distance)
            .to(dtype=dtype, device=device)
            .repeat(bn, 1, 1)
        )

        alibi_biases[cache_key] = buffered

    b = buffered[:target_size, :time_steps, :time_steps]
    b = b.view(batch_size, heads, time_steps, time_steps)
    return b

    
class AltAttention(nn.Module):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=True)
        # self.proj_drop = nn.Dropout(dropout)

    def forward(self, inputs, mask=None, alibi_bias=None):
        qkv = self.qkv(inputs)
        qkv = qkv.view(-1, inputs.shape[1], self.num_heads, self.dim * 3 // self.num_heads).permute(0, 2, 1, 3)
        q, k, v = qkv.split([self.dim // self.num_heads] * 3, dim=-1)

        if mask is not None:
            mask = mask[:, None, None, :]

        attn = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.scale

        if alibi_bias is not None:
            attn = attn.type_as(alibi_bias)
            attn += alibi_bias
            
        attn = MaskedSoftmax(dim=-1)(attn, mask=mask)#.to(q.dtype)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.permute(0, 2, 1, 3).reshape(-1, inputs.shape[1], self.dim)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x

class AltBlock(nn.Module):
    def __init__(self, dim=256, num_heads=4, expand=4, attn_dropout=0.2, mlp_dropout=0.2, drop_path=0., activation='gelu', prenorm=True, **kwargs):
        super().__init__(**kwargs)

        self.norm1 = nn.LayerNorm(dim)#MaskedBatchNorm1d(dim, momentum=0.05, channels_last=True)
        self.self_attn = AltAttention(dim=dim,num_heads=num_heads,dropout=attn_dropout)
        self.drop1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)#MaskedBatchNorm1d(dim, momentum=0.05, channels_last=True)
        self.mlp = GLUMlp(dim, expand, dropout=mlp_dropout, activation=activation)
        self.drop2 = DropPath(drop_path)

        self.prenorm = prenorm
        self.attn_scale = ScaleBiasLayer(dim, adaptive_scale=True)
        self.mlp_scale = ScaleBiasLayer(dim, adaptive_scale=True)
        
    def forward(self, inputs, mask=None, alibi_bias=None):
        x = inputs
        if self.prenorm:
            x = self.norm1(x)
        x = self.self_attn(x,mask=mask,alibi_bias=alibi_bias)
        x = self.drop1(x)
        x = self.attn_scale(x)
        x = x + inputs
        if not self.prenorm:
            x = self.norm1(x)
        attn_out = x

        if self.prenorm:
            x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop2(x)
        x = self.mlp_scale(x)
        x = x + attn_out
        if not self.prenorm:
            x = self.norm2(x)
        return x
        
class Conv1DBlock(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size=17,
                 groups=4,
                 dilation=1,
                 stride=1,
                 conv_dropout=0.0,
                 mlp_dropout=0.0,
                 drop_path=0.0,
                 expand=4,
                 activation='swish',
                 prenorm=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.prenorm = prenorm
        self.stride = stride

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.glu = GLU(dim=-1, activation=activation)
        self.expand_conv = nn.Linear(dim, 2*dim)
        self.conv = MaskedConv1d(dim, dim, kernel_size=kernel_size, groups=groups)
        self.conv_norm = MaskedBatchNorm1d(dim, momentum=0.05)
        self.conv_act = get_act_fn(activation)
        self.conv_proj = nn.Linear(dim, dim)
        self.mlp = GLUMlp(dim, expand, mlp_dropout, activation=activation)
        self.conv_dropout = nn.Dropout(conv_dropout)
        # self.mlp_dropout = nn.Dropout(mlp_dropout)
        self.drop1 = DropPath(drop_path)
        self.drop2 = DropPath(drop_path)
        self.conv_scale = ScaleBiasLayer(dim, adaptive_scale=True)
        self.mlp_scale = ScaleBiasLayer(dim, adaptive_scale=True)

    def compute_mask(self, inputs, mask=None):
      if mask is not None:
        if self.stride > 1:
          mask = mask[:,::self.stride]
      return mask

    def forward(self, inputs, mask=None):
        x = inputs
        if self.prenorm:
            x = self.norm1(x)
        x = self.expand_conv(x)
        x = self.glu(x)
        x = x.permute(0,2,1)
        x = self.conv(x,mask=mask)
        mask = self.compute_mask(inputs,mask)
        x = self.conv_norm(x,mask=mask)
        x = self.conv_act(x)
        x = self.conv_dropout(x)
        x = x.permute(0,2,1)
        x = self.conv_proj(x)
        x = self.drop1(x)
        x = self.conv_scale(x)
        if self.stride == 1:
            x = x + inputs
        if not self.prenorm:
            x = self.norm1(x)

        conv_out = x
        if self.prenorm:
            x = self.norm2(x)
        x = self.mlp(x)
        # x = self.mlp_dropout(x)
        x = self.drop2(x)
        x = self.mlp_scale(x)
        if self.stride == 1:
            x = x + conv_out
        if not self.prenorm:
            x = self.norm2(x)
        return x