from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.submodules.layer_modules import DropPath, ScaleBiasLayer
from models.submodules.masked_batchnorm import MaskedBatchNorm1d
from models.submodules.masked_conv import MaskedConv1d

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