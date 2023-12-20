import pandas as pd
import torch
import sys
import os
import argparse
from training import train_fold

class CFG:
    seed = 2023
    n_splits = 5
    fold = 0
    use_all_train = False #use all train data or not, fold will be ignored when True.
    max_len = -1 #input length will never exceed this value, -1 for unbounded.
    dynamic_len = True #apply dynamic padding on batch or not. If False, input length will be fixed to max_len(max_len must be > 0 for this case).
    mixed_precision = True

    world_size = 1
    grad_accumulation = 1
    clip_grad = 3.0
    lr = 2e-3 #for stability, 4e-3 is slightly better
    weight_decay = 0.01
    epochs = 60 
    warmup = 0.1
    train_batch_size = 256 // world_size
    valid_batch_size = train_batch_size
    neptune_project = "common/quickstarts"
    train_signal_to_noise_filter = 1.0 #signal_to_noise>this value

    dim = 192
    num_layers = 12
    num_heads = 4
    augment = False
    # awp = False

    # awp_start_epoch = epochs//10
    # awp_lr = 0.01
    # awp_eps = 0

    root_dir = '.'
    data_dir = '.'
    output_dir = './model_ckpts'
    model_path = ''
    
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
        if type(v) in [int, str, bool]:
            parser.add_argument(f'--{k}', type=type(v))
    return parser.parse_args()
    
if __name__ == '__main__':
    torch.distributed.init_process_group(backend="nccl", )
    rank = int(os.environ["LOCAL_RANK"])
    world_size = CFG.world_size#torch.distributed.get_world_size()
    device = "cuda:%s" % rank
    print(device)

    args = parse_args(CFG)
    update_cfg_from_args(CFG, args)
    sys.path.append(CFG.root_dir)
    if CFG.model_path == '':
        fold_name = CFG.fold if not CFG.use_all_train else 'all'
        CFG.model_path = f'squeeze-{CFG.dim}-{CFG.num_layers}-{CFG.num_heads}-snfilter{CFG.train_signal_to_noise_filter}-ep{CFG.epochs}-fold{fold_name}-seed{CFG.seed}.pth'
    if rank == 0:
        print(CFG.model_path)

    train_df = pd.read_parquet(f'{CFG.data_dir}/train_data_new.parquet')
    train_fold(CFG, train_df, device, rank, world_size)