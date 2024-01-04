import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.submodules.modules import Conv1DBlock,  AltBlock
from models.submodules.alibi import get_alibi
from models.submodules.masked_conv import MaskedConv2d

class SqueezeformerBlock(nn.Module):
    def __init__(self,
                 dim=256,
                 kernel_size=17,
                 groups=4,
                 num_heads=4,
                 conv_expand=4,
                 attn_expand=4,
                 num_conv_block=1,
                 num_attn_block=1,
                 conv_dropout=0.1,
                 attn_dropout=0.1,
                 mlp_dropout=0.1,
                 drop_path=0.1,
                 activation='swish',
                 prenorm=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.conv_blocks = nn.ModuleList([Conv1DBlock(dim,kernel_size,groups,1,1,conv_dropout,mlp_dropout,drop_path,conv_expand,activation,prenorm) for _ in range(num_conv_block)])
        self.attn_blocks = nn.ModuleList([AltBlock(dim,num_heads,attn_expand,attn_dropout,mlp_dropout,drop_path,activation,prenorm) for _ in range(num_attn_block)])

    def forward(self, inputs, mask=None, alibi_bias=None):
        x = inputs #(B,N,C)
        for block in self.conv_blocks:
            x = block(x, mask=mask)
        for block in self.attn_blocks:
            x = block(x, mask=mask, alibi_bias=alibi_bias)
        return x

class MaskedRNN(nn.Module):
    def __init__(self, rnn_module, **kwargs):
        super().__init__(**kwargs)
        self.rnn_module = rnn_module
        assert rnn_module.batch_first
        
    def forward(self, inputs, mask=None):
        if mask is not None:
            orig_len = inputs.size(1)
            x = inputs * mask.to(inputs.dtype)[:,:,None]
            lens = mask.to(torch.int32).sum(1).to('cpu')
            x = nn.utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=True, enforce_sorted=False)
            x, hidden = self.rnn_module(x)
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            x = F.pad(x, (0,0,0,orig_len-x.size(1)))
        else:
            x, hidden = self.rnn_module(inputs)
        return x, hidden

class BPPConvnet(nn.Module):
    def __init__(self, out_ch=4, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = MaskedConv2d(1,64,kernel_size=7,bias=True)
        self.act1 = nn.ReLU()
        self.conv2 = MaskedConv2d(64,out_ch,kernel_size=7,bias=True)
        self.final_act = nn.Identity()

    def forward(self, bpp, mask=None):
        x = bpp
        x = self.conv1(x,mask)
        x = self.act1(x)
        x = self.conv2(x,mask)
        x = self.final_act(x)
        return x
        
class RNAEncoder(nn.Module):
    def __init__(self,
                 dim=384,
                 kernel_size=17,
                 groups=4,
                 num_heads=4,
                 num_layers=12,
                 conv_expand=4,
                 attn_expand=4,
                 num_conv_block=1,
                 num_attn_block=1,
                 conv_dropout=0.1,
                 attn_dropout=0.1,
                 mlp_dropout=0.1,
                 drop_path=0.1,
                 activation='swish',
                 prenorm=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.embedding = nn.Embedding(5,dim)
        self.prenorm = prenorm
        self.mfe_embedding = nn.Embedding(4,dim)
        self.looptype_embedding = nn.Embedding(8,dim)
        self.capr_looptype_embedding = nn.Linear(6, dim)
        self.emb_proj = nn.Linear(5*dim, dim)
        self.emb_dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(dim)
        # self.alibi_bias = nn.Parameter(get_alibi(512, num_heads), requires_grad=False)
        self.bpp_feature_proj = nn.Linear(3, dim)
        self.bpp_convnet = BPPConvnet(num_heads)
        self.num_heads = num_heads
        self.layers = nn.ModuleList(
                    [SqueezeformerBlock(dim,kernel_size,groups,num_heads,conv_expand,attn_expand,num_conv_block,num_attn_block,conv_dropout,attn_dropout,mlp_dropout,drop_path,activation,prenorm) for _ in range(num_layers)])
    
    def forward(self, inp, mask=None):
        x1 = self.embedding(inp['seq'])
        x2 = self.mfe_embedding(inp['mfe'])
        x3 = self.bpp_feature_proj(inp['bpp_features'])
        x4 = self.looptype_embedding(inp['loop_type'])
        x5 = self.capr_looptype_embedding(inp['capr_loop_type'])

        x = torch.cat([x1, x2, x3, x4, x5], dim=-1)
        x = self.emb_dropout(x)
        x = self.emb_proj(x)
        x = self.emb_dropout(x)
        x = self.norm(x)

        bpp = self.bpp_convnet(inp['bpp'].unsqueeze(1))
        alibi_bias = get_alibi(x.size(1), self.num_heads).to(dtype=x.dtype, device=x.device).repeat(x.size(0), 1, 1, 1)
        alibi_bias = alibi_bias + bpp

        outputs = []
        for layer in self.layers:
            x = layer(x, mask=mask, alibi_bias=alibi_bias)
            if mask is not None and hasattr(layer, 'compute_mask'):
                mask = layer.compute_mask(x, mask)

            outputs.append(x)

        return outputs
        
class RNARegModel(nn.Module):
    def __init__(self, dim=192, num_layers=12, num_heads=4,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = RNAEncoder(dim, num_layers=num_layers, groups=dim, num_heads=num_heads)
        self.head_dropout = nn.Dropout(0.5)
        self.rnn = nn.GRU(dim, dim, num_layers=1, batch_first=True, bidirectional=False)
        self.reg_head = nn.ModuleList([
                        # nn.Dropout(0.5),
                        nn.Linear(dim,2),
        ])
                      

    def forward(self, input):
        mask = input['mask']
        x = self.encoder(input, mask)[-1]
        x = self.head_dropout(x)
        x, _ = self.rnn(x)
        react_pred = x
        for layer in self.reg_head:
            react_pred = layer(react_pred)

        output = {}
        output['react_pred'] = react_pred
            
        return output