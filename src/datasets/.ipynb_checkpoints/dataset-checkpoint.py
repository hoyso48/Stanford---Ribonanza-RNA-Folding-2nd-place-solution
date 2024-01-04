import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from typing import List, Tuple, Union

def convert(string, types):
    return tuple(typ(entry) for typ, entry in zip(types, string.split()))

def read_bpp(path):
    rows = []
    cols = []
    values = []
    with open(path) as f:
        for line in f.readlines():
            try:
                r, c, v = convert(line, (int, int, float))
                rows.append(r-1)
                cols.append(c-1)
                values.append(v)
            except:
                break
    return [rows, cols], values

def parse_capr(file_path):
    """
    Parses the given file, adjusting for the specific file structure (ignoring the header line),
    and returns a dictionary with keys corresponding to the labels (e.g., 'Bulge', 'Exterior', etc.)
    and values as lists of floats.

    :param file_path: Path to the file to be parsed.
    :return: Dictionary with keys as labels and values as lists of floats.
    """
    result_dict = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Skip the first line which is a header
        for line in lines[1:]:
            parts = line.split()
            if parts:  # Check if line is not empty
                key = parts[0]
                values = [float(value) for value in parts[1:]]
                result_dict[key] = values

    return result_dict

class RNADataset(Dataset):
    def __init__(self, df, mode='train', SN_filter=True, dtype='float32', signal_to_noise_filter=None, dir='.'):
        if SN_filter:
            m = (df['2A3_SN_filter'].values > 0) & (df['DMS_SN_filter'].values > 0)
            df = df.loc[m].reset_index(drop=True)
        if signal_to_noise_filter:
            m = (df['2A3_signal_to_noise'].values > signal_to_noise_filter) & (df['DMS_signal_to_noise'].values > signal_to_noise_filter)
            df = df.loc[m].reset_index(drop=True)
        self.seq_map = {'A':1,'C':2,'G':3,'U':4}
        self.mfe_map = {'.':1, '(':2, ')':3}
        self.loop_type_map = {'B':1, 'E':2, 'H':3, 'I':4, 'M':5, 'S':6, 'X':7}
        self.seq = df['sequence'].values
        self.mfe = df['mfe_eternafold'].values
        self.loop_type = df['looptype_eternafold'].values
        if mode != 'test':
            self.react_2A3 = df[[c for c in df.columns if \
                                     '2A3_reactivity_0' in c]].values.astype(dtype)
            self.react_DMS = df[[c for c in df.columns if \
                                     'DMS_reactivity_0' in c]].values.astype(dtype)
            self.react_err_2A3 = df[[c for c in df.columns if \
                                     '2A3_reactivity_error_0' in c]].values.astype(dtype)
            self.react_err_DMS = df[[c for c in df.columns if \
                                    'DMS_reactivity_error_0' in c]].values.astype(dtype)
            self.sn_2A3 = df['2A3_signal_to_noise'].values.astype(dtype)
            self.sn_DMS = df['DMS_signal_to_noise'].values.astype(dtype)
        
        self.bpp_paths = df['bpp_path'].values
        self.dir = dir
        self.mode = mode
        del df
    def __len__(self):
        return len(self.seq)
    def __getitem__(self, idx):

        seq = self.seq[idx]
        mfe = self.mfe[idx]
        loop_type = self.loop_type[idx]

        seq_len = len(seq)

        mask = torch.ones(seq_len, dtype=torch.bool)
        
        seq = np.array([self.seq_map[s] for s in seq])
        mfe = np.array([self.mfe_map[s] for s in mfe])
        loop_type = np.array([self.loop_type_map[s] for s in loop_type])
        seq = torch.from_numpy(seq)
        mfe = torch.from_numpy(mfe)
        loop_type = torch.from_numpy(loop_type)

        bpp = torch.sparse_coo_tensor(*read_bpp(f'{self.dir}/Ribonanza_bpp_files/extra_data/{self.bpp_paths[idx]}.txt'), size=(seq_len,seq_len)).to_dense()
        # bpp = bpp + torch.eye(self.max_len)
        bpp = bpp + bpp.T
        bpp_sum = bpp.sum(-1)
        bpp_nzero = (bpp == 0).float().sum(-1) / seq_len
        bpp_max = bpp.max(dim=-1)[0]
        bpp_features = torch.stack([bpp_sum, bpp_nzero, bpp_max], dim=-1)

        capr_loop_type = parse_capr(f'{self.dir}/CapR/CapR_predictions/{self.bpp_paths[idx]}.txt')
        capr_loop_type = np.stack([capr_loop_type['Bulge'], capr_loop_type['Exterior'], capr_loop_type['Hairpin'], capr_loop_type['Internal'], capr_loop_type['Multibranch'], capr_loop_type['Stem']], -1)
        capr_loop_type = torch.from_numpy(capr_loop_type).float()

        if self.mode != 'test':
            react = np.stack([self.react_2A3[idx][:seq_len],
                             self.react_DMS[idx][:seq_len]],-1)
            react = torch.from_numpy(react)
            react_err = np.stack([self.react_err_2A3[idx][:seq_len],
                             self.react_err_DMS[idx][:seq_len]],-1)
            react_err = torch.from_numpy(react_err)
            snr = np.array([self.sn_2A3[idx], self.sn_DMS[idx]])
            snr = torch.from_numpy(snr)
        else:
            react = torch.full((seq_len,2), float('nan'), dtype=torch.float32)
            react_err = torch.full((seq_len,2), float('nan'), dtype=torch.float32)
            snr = torch.full((2,), float('nan'), dtype=torch.float32)
        
        return {'seq':seq, 'mfe':mfe, 'loop_type':loop_type, 'capr_loop_type':capr_loop_type, 'bpp_features':bpp_features, 'bpp':bpp, 'mask':mask}, {'react':react,  'react_err':react_err, 'mask':mask, 'snr':snr}

def trunc_pad_stack(
    inp: List[torch.Tensor], 
    dim: Union[int, List[int], Tuple[int, ...]] = 0, 
    max_len: Union[int, List[int], Tuple[int, ...]] = -1, 
    dynamic: Union[bool, List[bool], Tuple[bool, ...]] = True,
    padding_value: Union[int, str, float] = 0,
    stack_dim: int = 0,
    position: str = 'post',  # 'post', 'pre'
    mode: str = 'constant',  # 'constant', 'reflect', 'replicate', 'circular'
) -> torch.Tensor:

    if len(inp) == 0:
        return torch.empty(0)

    if isinstance(dim, int):
        dim = [dim]
    elif isinstance(dim, tuple):
        dim = list(dim)

    if len(dim) == 0:
        return torch.stack(inp, dim=stack_dim)
        
    if isinstance(max_len, int):
        max_len = [max_len] * len(dim)
    elif isinstance(max_len, tuple):
        max_len = list(max_len)
        
    if len(dim) != len(max_len):
        raise ValueError("When max_len is a list or tuple, its length must match that of dim.")

    if isinstance(dynamic, bool):
        dynamic = [dynamic] * len(dim)
    elif isinstance(dynamic, tuple):
        dynamic = list(dynamic)

    if len(dim) != len(dynamic):
        raise ValueError("When dynamic is a list or tuple, its length must match that of dim.")

    _slice = [slice(None)] * len(inp[0].shape)
    for d, m_len in zip(dim, max_len):
        if m_len == -1:
            # if max_len == -1, do not truncate
            _slice[d] = slice(None, None)
        else:
            _slice[d] = slice(None, m_len)

    for i in range(len(inp)):
        inp[i] = inp[i][_slice]

    _size = [x.size() for x in inp]

    max_lens = list(_size[0])
    for d, m_len, dy in zip(dim, max_len, dynamic):
        assert not (not dy and m_len == -1)
        if dy or m_len==-1:
            max_lens[d] = max([x[d] for x in _size])
        else:
            max_lens[d] = m_len
    
    for i in range(len(inp)):
        paddings = [(0,0)] * len(inp[0].shape)
        if position == 'pre':
            for d, m_len in zip(dim, max_len):
                paddings[d] = (max_lens[d] - _size[i][d], 0)
        elif position == 'post':
            for d, m_len in zip(dim, max_len):
                paddings[d] = (0, max_lens[d] - _size[i][d])
        else:
            raise ValueError(f"Invalid padding position {position}")
        paddings.reverse()
        paddings = [x for xs in paddings for x in xs]
        inp[i] = F.pad(inp[i], paddings, mode=mode, value=padding_value)

    inp = torch.stack(inp, dim=stack_dim)
        
    return inp
    
def collate_fn(batch, max_len=-1, dynamic_len=True):
    # batch contains a list of tuples of structure (sequence, target)
    batch_x, batch_y = zip(*batch)
    b_x, b_y = {}, {}
    for key in batch_x[0].keys():
        if key == 'bpp':
            b_x[key] = trunc_pad_stack([x[key] for x in batch_x], dim=[0,1], max_len=max_len, dynamic=dynamic_len)
        else:
            b_x[key] = trunc_pad_stack([x[key] for x in batch_x], dim=0, max_len=max_len, dynamic=dynamic_len)

    for key in batch_y[0].keys():
        if key == 'snr':
            b_y[key] = torch.stack([x[key] for x in batch_y])
        else:
            b_y[key] = trunc_pad_stack([x[key] for x in batch_y], dim=0, max_len=max_len, dynamic=dynamic_len)
    return b_x, b_y
