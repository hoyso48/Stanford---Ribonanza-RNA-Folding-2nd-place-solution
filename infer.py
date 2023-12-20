import pandas as pd
import torch
import sys
import os
import glob
import argparse
from inference import make_submission

class CFG:
    max_len = -1
    dynamic_len = True
    mixed_precision = True

    device = 'cuda:0'
    dim = 192
    num_layers = 12
    num_heads = 4

    root_dir = '.'
    data_dir = '.'
    model_dir = './model_ckpts'
    filename = './submission.parquet'
    
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
    
if __name__ == '__main__':
    sys.path.append(CFG.root_dir)
    device = CFG.device
    print(device)

    args = parse_args(CFG)
    update_cfg_from_args(CFG, args)
    sys.path.append(CFG.root_dir)
    test_df = pd.read_parquet(f'{CFG.data_dir}/test_sequences_new.parquet')
    model_paths = glob.glob(f'{model_dir}/*.pth')
    make_submission(CFG, model_paths, test_df, CFG.filename)