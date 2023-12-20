import gc
import torch
import ctypes
import neptune
import random
import numpy as np
import os
import glob

os.environ['NEPTUNE_API_TOKEN'] = neptune.ANONYMOUS_API_TOKEN #place your neptune token

# Seed all random number generators
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def init_everything(cfg):
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()
    seed_everything(cfg.seed)

def init_neptune(cfg):
    #init neptune
    if cfg.neptune_project == "common/quickstarts":
        neptune_api_token=neptune.ANONYMOUS_API_TOKEN
    else:
        neptune_api_token=os.environ['NEPTUNE_API_TOKEN']
        
    fns = glob.glob(f"*.py")
    
    neptune_run = neptune.init_run(
            project=cfg.neptune_project,
            tags="demo",
            mode="async",
            api_token=neptune_api_token,
            capture_stdout=False,
            capture_stderr=False,
            source_files=fns,
            capture_hardware_metrics=True
        )
    # print(f"Neptune system id : {neptune_run._sys_id}")
    # print(f"Neptune URL       : {neptune_run.get_url()}")
    neptune_run["cfg"] = neptune.utils.stringify_unsupported({k: v for k, v in cfg.__dict__.items() if not k.startswith('__') and not k.endswith('__')})
    return neptune_run

def neptune_log_const(run, key, value, step, nan_replace_value=10000, warn_nans=True):
    if isinstance(value, torch.Tensor):
        value = value.item()
    if np.isinf(value):
        if warn_nans:
            print(f'[WARNING] Inf produced in value {key}, replaced with {nan_replace_value} in neptune log')
        value = nan_replace_value
    if np.isnan(value):
        if warn_nans:
            print(f'[WARNING] NaN produced in value {key}, replaced with {nan_replace_value} in neptune log')
        value = nan_replace_value
    return run[key].log(value=value, step=step)