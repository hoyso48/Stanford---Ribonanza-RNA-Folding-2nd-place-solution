import pandas as pd
import numpy as np
import gc
import os
import sys
import glob
import argparse

import torch
from fastai.vision.all import DataLoader
from datasets.dataset import RNADataset, collate_fn
from models.model import RNARegModel
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

class CFG:
    max_len = -1
    dynamic_len = True
    mixed_precision = True

    device = 'cuda:0'
    num_workers = 8
    batch_size = 256
    dim = 192
    num_layers = 12
    num_heads = 4

    exist_skip = True
    df_name = 'test_sequences_new.parquet'
    pq_suffix = '-sub'
    root_dir = '.'
    data_dir = '../datamount'
    output_dir = '../outputs'
    model_dir = 'model_ckpts'
    filename = 'submission.parquet'

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
    
def get_predictions(model, df, cfg):
    ds = RNADataset(df, mode='test', SN_filter=False, dir=cfg.data_dir)
    dl = DataLoader(ds,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    sampler=None,
                    num_workers=cfg.num_workers,
                    persistent_workers=True,
                    drop_last=False,
                    pin_memory=True,
                    prefetch_factor=2,
                    create_batch=lambda x: collate_fn(x, cfg.max_len, cfg.dynamic_len),
                   )
    device = cfg.device
    model = model.to(device)
    model.eval()
    _iterator = iter(dl)
    predictions = []
    with torch.no_grad():
        for step in tqdm(range(len(dl)), desc=f'predicting'):
            data, _ = next(_iterator)
            data = dict([(k, v.to(device)) for k, v in data.items()])
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                result = model(data)
            predictions.append(result['react_pred'])
    return predictions

def predictions_to_sub(predictions, df):
    preds = [x.detach().cpu().float().numpy() for x in predictions]
    preds = [item for sublist in preds for item in sublist]
    preds = np.concatenate([x[:l] for x,l in zip(preds, df.seq_len.values)])
    # assert len(preds) == 269796671
    sub_df = pd.DataFrame({'id':np.arange(0, len(preds), 1), 
                               'reactivity_DMS_MaP':preds[:,1], 
                               'reactivity_2A3_MaP':preds[:,0]})
    return sub_df

# def predictions_to_df(predictions, df, max_len=206):
#     preds = [x.detach().cpu().float().numpy() for x in predictions]
#     preds = [item for sublist in preds for item in sublist]
#     preds = [x[:min(l,max_len)] for x,l in zip(preds, df.seq_len.values)]
#     preds = np.stack([np.pad(x, ((0,max_len-len(x)),(0,0)), mode='constant', constant_values=np.nan) for x in preds])
#     for i in range(max_len):
#         df.loc[:,[f'2A3_reactivity_{i+1:04d}']] = preds[:,i,0]
#     for i in range(max_len):
#         df.loc[:,[f'DMS_reactivity_{i+1:04d}']] = preds[:,i,1]
#     return df

def merge_submissions(sub_dfs, clip=True, weights=None, plot=True):
    if weights is None:
        N = len(sub_dfs)
        weights = [1./N for _ in sub_dfs]
    else:
        W = sum(weights)
        weights = [w/W for w in weights]
    print('weights:',weights)
    init = False
    for p, w in zip(sub_dfs, weights):
        df = pd.read_parquet(p)
        if not init:
            df_len = len(df)
            p_DMS = np.zeros((df_len,), dtype=np.float32)
            p_2A3 = np.zeros((df_len,), dtype=np.float32)
            init = True
        if plot:
            plt.plot(df.iloc[-457:-150,1:3])
            plt.show()
        
        p_DMS += df['reactivity_DMS_MaP'].values * w
        p_2A3 += df['reactivity_2A3_MaP'].values * w
        del df
    if clip:
        p_DMS = np.clip(p_DMS, 0, 1)
        p_2A3 = np.clip(p_2A3, 0, 1)
    sub_df = pd.DataFrame({'id':np.arange(0, df_len, 1), 
                               'reactivity_DMS_MaP':p_DMS, 
                               'reactivity_2A3_MaP':p_2A3})
    if plot:
        plt.plot(sub_df.iloc[-457:-150,1:3])
        plt.show()
    return sub_df

def make_submission(cfg, model_paths, df, filename='submission.parquet', weights=None, clip=True, exist_skip=True, output_dir='.', pq_suffix=''):
    fns = []
    for p in model_paths:
        fname = p.split('/')[-1][:-4]
        fn = f'{output_dir}/{fname}{pq_suffix}.parquet'
        if exist_skip and os.path.isfile(fn):
            print(f'file {fn} already exists, skip prediction')
            fns.append(fn)
        else:
            loaded = False
            model = RNARegModel(cfg.dim, cfg.num_layers, cfg.num_heads)
            loaded_state_dict = torch.load(f'{p}')['model']
            model.load_state_dict(loaded_state_dict, strict=True)
            model = torch.compile(model)
            print(f"loaded weights from:{p}")
                
            model.eval()
            predictions = get_predictions(model, df, cfg)
            sub = predictions_to_sub(predictions, df)
            print(f'wrote {fn}')
            sub.to_parquet(fn)
            fns.append(fn)
            del sub, predictions, model
            gc.collect()
    sub = merge_submissions(fns, clip=True, weights=weights, plot=True)
    sub.to_parquet(filename)
    print('submission file created:', filename) 
    return sub


def infer(cfg=CFG):
    os.chdir(cfg.root_dir)
    test_df = pd.read_parquet(f'{cfg.data_dir}/{cfg.df_name}')
    model_paths = glob.glob(f'{cfg.output_dir}/{cfg.model_dir}/*.pth')
    print('get predictions from:', model_paths)
    filename = f'{cfg.output_dir}/{CFG.filename}'
    make_submission(cfg, model_paths, test_df, filename, exist_skip=CFG.exist_skip, output_dir=CFG.output_dir, pq_suffix=CFG.pq_suffix)

if __name__ == '__main__':
    args = parse_args(CFG)
    update_cfg_from_args(CFG, args)
    infer(CFG)
