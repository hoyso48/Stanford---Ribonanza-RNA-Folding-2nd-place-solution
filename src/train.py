from models.model import RNARegModel
from datasets.dataset import RNADataset, collate_fn
from utils.util import seed_everything, clear_everything
from callbacks.callback import SaveLastCallback
from losses.loss import mae_loss, weighted_mae_loss
from metrics.metric import MAE, MAE_ave

from fastai.vision.all import GradientClip, DataLoaders, DataLoader, Metric, Learner, Adam, CSVLogger
from fastai.distributed import *

import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

from sklearn.model_selection import KFold
from tqdm.autonotebook import tqdm
import gc
import os
from functools import partial
import argparse

class CFG:
    seed = 2023
    n_splits = 5
    fold = 0
    use_all_train = False #use all train data or not, fold will be ignored when True.
    max_len = -1 #input length will never exceed this value, -1 for unbounded. set it to 206 when num_processes>1
    dynamic_len = True #apply dynamic padding on batch or not. If False, input length will be fixed to max_len(max_len must be > 0 for this case). set it to False when num_processes>1
    
    mixed_precision = True

    num_workers = 8
    device = 'cuda'
    # grad_accumulation = 1
    clip_grad = 3.0
    lr = 2e-3 #for stability. 4e-3 is slightly better in scores.
    weight_decay = 0.01
    epoch = 60 
    warmup = 0.01
    train_batch_size = 256
    valid_batch_size = train_batch_size
    valid_drop_last = False #set it to True when num_processes>1
    train_signal_to_noise_filter = 1.0 #signal_to_noise>this value

    dim = 192
    num_layers = 12
    num_heads = 4
    
    root_dir = '.'
    data_dir = '../datamount'
    output_dir = '../outputs'
    model_dir = 'model_ckpts'
    model_name = 'model'

def update_cfg_from_args(cfg, args):
    for key, value in vars(args).items():
        if hasattr(cfg, key):
            if value is not None:
                setattr(cfg, key, value)
        else:
            raise ValueError(f"Unknown argument: {key}")

def parse_args(cfg):
    parser = argparse.ArgumentParser()
    cfg_dict = {k: v for k, v in cfg.__dict__.items() if not k.startswith('__') and not k.endswith('__')}
    for k, v in cfg_dict.items():
        parser.add_argument(f'--{k}', type=type(v) if v is not None else int)
    return parser.parse_args()
    
def train(cfg=CFG):
    os.chdir(cfg.root_dir)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f'{cfg.output_dir}/{cfg.model_dir}', exist_ok=True)
    seed_everything(cfg.seed)
    train_df = pd.read_parquet(f'{cfg.data_dir}/train_data_new.parquet')#.sample(frac=0.01).reset_index(drop=True)

    df = train_df
    if cfg.use_all_train:
        train = df
        valid = df[:0]
    else:
        df['fold'] = -1
        df = df.reset_index(drop=True)
    
        kfold = KFold(n_splits=cfg.n_splits, random_state=cfg.seed, shuffle=True)
        for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(df)):
            df.loc[valid_idx, 'fold'] = fold_idx
    
        assert not (df['fold']==-1).sum()
        assert len(np.unique(df['fold']))==cfg.n_splits
    
        train = df.loc[df['fold']!=cfg.fold]
        valid = df.loc[df['fold']==cfg.fold]

    train_ds = RNADataset(train, mode='train', SN_filter=False, signal_to_noise_filter=cfg.train_signal_to_noise_filter, dir=cfg.data_dir)
    valid_ds = RNADataset(valid, mode='valid', SN_filter=True, dir=cfg.data_dir)
    train_loader = DataLoader(train_ds,
                        batch_size=cfg.train_batch_size,
                        shuffle=True,
                        sampler=None,
                        num_workers=cfg.num_workers,
                        persistent_workers=True,
                        drop_last=True,
                        pin_memory=True,
                        prefetch_factor=2,
                        create_batch=lambda x: collate_fn(x, cfg.max_len, cfg.dynamic_len),
                       )
    valid_loader = DataLoader(valid_ds,
                        batch_size=cfg.valid_batch_size,
                        shuffle=False,
                        sampler=None,
                        num_workers=cfg.num_workers,
                        persistent_workers=True,
                        drop_last=cfg.valid_drop_last,
                        pin_memory=True,
                        prefetch_factor=2,
                        create_batch=lambda x: collate_fn(x, cfg.max_len, cfg.dynamic_len),
                       )
    model = RNARegModel(dim=cfg.dim, num_layers=cfg.num_layers, num_heads=cfg.num_heads)
    model = torch.compile(model.to(cfg.device))
    data = DataLoaders(train_loader, valid_loader)
    clear_everything()

    learn = Learner(data, 
                model, 
                loss_func=weighted_mae_loss,
                path=cfg.output_dir,
                model_dir=cfg.model_dir,
                cbs=[GradientClip(cfg.clip_grad), SaveLastCallback(cfg.model_name), CSVLogger(fname=f'{cfg.model_name}-log.csv')],
                opt_func=partial(Adam, lr=cfg.lr, mom=0.9, sqr_mom=0.999, eps=1e-8),
            metrics=[MAE_ave()]).to_fp16(enabled=cfg.mixed_precision)

    with learn.distrib_ctx(sync_bn=False): #masked sync_bn is not implemented

        learn.fit_one_cycle(cfg.epoch,
                            lr_max=cfg.lr,
                            wd=cfg.weight_decay,
                            pct_start=cfg.warmup)
    return learn


if __name__ == '__main__':
    from accelerate.utils import write_basic_config
    write_basic_config()
    args = parse_args(CFG)
    update_cfg_from_args(CFG, args)
    train(CFG)
