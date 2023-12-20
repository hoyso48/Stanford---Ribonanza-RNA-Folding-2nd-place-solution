import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

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
        self.loop_type = df['loop_type_eternafold'].values
        if mode != 'test':
            self.react_2A3 = df[[c for c in df.columns if \
                                     '2A3_reactivity_0' in c]].values.astype(dtype)
            self.react_DMS = df[[c for c in df.columns if \
                                     'DMS_reactivity_0' in c]].values.astype(dtype)
            # self.react_err_2A3 = df[[c for c in df_2A3.columns if \
            #                          '2A3_reactivity_error_0' in c]].values.astype(dtype)
            # self.react_err_DMS = df[[c for c in df_DMS.columns if \
            #                         'DMS_reactivity_error_0' in c]].values.astype(dtype)
            self.sn_2A3 = df['2A3_signal_to_noise'].values
            self.sn_DMS = df['DMS_signal_to_noise'].values
        
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
            snr = np.array([self.sn_2A3[idx], self.sn_DMS[idx]])
            snr = torch.from_numpy(snr)
        else:
            react = torch.full((seq_len,2), float('nan'), dtype=torch.float32)
            snr = torch.full((2,), float('nan'), dtype=torch.float32)
        
        return {'seq':seq, 'mfe':mfe, 'loop_type':loop_type, 'capr_loop_type':capr_loop_type, 'bpp_features':bpp_features, 'bpp':bpp,  'react':react,  'mask':mask, 'snr':snr}
        
def pad_seq(seq, max_len=-1, dynamic_len=True):
    if max_len != -1:
        seq = [x[:max_len] for x in seq]
    if dynamic_len:
        max_len = max([len(x) for x in seq])
    padded = []
    for x in seq:
        padded.append(F.pad(x,(0,max_len-len(x)),'constant',0))
    return torch.stack(padded)

def pad_tar(seq, max_len=-1, dynamic_len=True):
    if max_len != -1:
        seq = [x[:max_len] for x in seq]
    if dynamic_len:
        max_len = max([len(x) for x in seq])
    padded = []
    for x in seq:
        padded.append(F.pad(x,(0,0,0,max_len-len(x)),'constant',0))
    return torch.stack(padded)

def pad_bpp(bpp, max_len=-1, dynamic_len=True):
    if max_len != -1:
        bpp = [x[:max_len,:max_len] for x in bpp]
    if dynamic_len:
        max_len = max([len(x) for x in bpp])
    padded = []
    for x in bpp:
        padded.append(F.pad(x,(0,max_len-len(x),0,max_len-len(x)),'constant',0))
    return torch.stack(padded)

def random_flip(x, p=0.5):
    if np.random.uniform() <p:
        x = x.flip(0)
    return x

def flip_batch(batch, p=0.5):
    return [dict([(k,random_flip(v, p=p)) for k, v in x.items()]) for x in batch]

def collate_fn(batch, max_len=-1, dynamic_len=True, augment=False):
    # batch contains a list of tuples of structure (sequence, target)
    # if augment:
    #     batch = flip_batch(batch, p=0.5)
    b = {}
    b['seq'] = pad_seq([x['seq'] for x in batch], max_len, dynamic_len)
    b['mfe'] = pad_seq([x['mfe'] for x in batch], max_len, dynamic_len)
    b['loop_type'] = pad_seq([x['loop_type'] for x in batch], max_len, dynamic_len)
    b['mask'] = pad_seq([x['mask'] for x in batch], max_len, dynamic_len)
    b['react'] = pad_tar([x['react'] for x in batch], max_len, dynamic_len)
    b['capr_loop_type'] = pad_tar([x['capr_loop_type'] for x in batch], max_len, dynamic_len)
    b['bpp_features'] = pad_tar([x['bpp_features'] for x in batch], max_len, dynamic_len)
    b['bpp'] = pad_bpp([x['bpp'] for x in batch], max_len, dynamic_len)
    b['snr'] = torch.stack([x['snr'] for x in batch])
    return b

def get_dataset(cfg, df, mode='train', batch_size=16, shuffle=False, num_workers=8, drop_last=False, pin_memory=True, seed=42, rank=0, world_size=1):
    if mode == 'train':
        augment = False
        SN_filter = False
        signal_to_noise_filter = cfg.train_signal_to_noise_filter
    elif mode == 'valid':
        augment = False
        SN_filter = True
        signal_to_noise_filter = None
    else:
        raise NotImplementedError

    max_len = cfg.max_len
    dynamic_len = cfg.dynamic_len
    dir = cfg.data_dir
    
    ds = RNADataset(df, mode=mode, SN_filter=SN_filter, signal_to_noise_filter=signal_to_noise_filter, dir=dir)

    sampler = None
    batch_sampler = None
    
    if world_size > 1:
        #dynamic shape is not supported for DDP
        if not drop_last:
            if rank == 0:
                print('force drop_last=True for DDP training')
            drop_last = True
        if max_len == -1:
            if rank == 0:
                print('force max_len=206 for DDP training')
            max_len = 206
        if dynamic_len == True:
            if rank == 0:
                print('force dynamic_len=False for DDP training')
            dynamic_len = False
        sampler = DistributedSampler(ds,
                                       num_replicas=world_size,
                                       rank=rank,
                                       shuffle=shuffle,
                                       seed=seed,
                                       drop_last=drop_last,
                                       )
        shuffle = False


    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        sampler=sampler,
                        batch_sampler=batch_sampler,
                        num_workers=num_workers,
                        persistent_workers=True,
                        drop_last=drop_last,
                        pin_memory=pin_memory,
                        prefetch_factor=2,
                        collate_fn=lambda x: collate_fn(x, max_len, dynamic_len, augment),
                       )
    del ds, df
    return loader
