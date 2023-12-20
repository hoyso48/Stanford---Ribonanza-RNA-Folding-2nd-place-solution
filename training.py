import pandas as pd
import numpy as np

import torch
from torch_optimizer import Lookahead, RAdam
from transformers import get_cosine_schedule_with_warmup
from torchinfo import summary

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from sklearn.model_selection import KFold
from tqdm.autonotebook import tqdm
import gc
import os

from utils import seed_everything, init_everything, init_neptune, neptune_log_const
from model import get_model
from data import get_dataset
from metrics import get_metrics
from awp import AWP

def step_fn(global_step, cfg, model, device, data, scaler, optimizer, scheduler, metrics_aggregator, rank=0, world_size=1, awp=None, mode='train'):
    data = dict([(k, v.to(device)) for k, v in data.items()])

    #forward pass
    def forward():
        with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
            result = model(data)
        return result

    result = forward()

    loss = 0
    for k, v in result['losses'].items():
        loss += v

    #update metrics
    for k, v in metrics_aggregator.items():
        v.update(result)

    if mode=='train':
        #backprop
        if cfg.grad_accumulation >1:
            loss /= cfg.grad_accumulation
            
        if awp is not None and awp.start_step <= global_step:
            scaler.scale(loss).backward()
            awp.on_retrain_begin(global_step)
            result = forward()

        scaler.scale(loss).backward()

        if awp is not None and awp.start_step <= global_step:
            awp.on_retrain_end(global_step)

        if global_step % cfg.grad_accumulation == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            result['grad_norm'] = grad_norm
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        scheduler.step()

    return result


def train_loop(cfg, train_files, valid_files, device, rank=0, world_size=1, model_summary=True):
    init_everything(cfg)
    if rank==0:
        os.makedirs(cfg.output_dir, exist_ok=True)
    #init neptune
    if rank==0:
        neptune_run = init_neptune(cfg)
    else:
        neptune_run = None
        
    #init dataset
    train_ds = get_dataset(cfg, train_files, mode='train', batch_size=cfg.train_batch_size, drop_last=True, shuffle=True, rank=rank, world_size=world_size)
    valid_ds = get_dataset(cfg, valid_files, mode='valid', batch_size=cfg.valid_batch_size, shuffle=False, rank=rank, world_size=world_size)
    num_train = len(train_ds.dataset)
    num_valid = len(valid_ds.dataset)
    steps_per_epoch = len(train_ds) #drop_remainder=True
    valid_steps_per_epoch = len(valid_ds) #drop_remainder=False

    global_step = 0

    model = get_model(cfg)
    
    if rank == 0:
        print(f'num_train:{num_train} num_valid:{num_valid}')
        if model_summary:
            for x in train_ds:
                temp_train = x
                break
            summary(model, input_data=[temp_train], verbose=1)
            del temp_train
            
    model = model.to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank], static_graph=False)
    model = torch.compile(model)
    
    metrics_aggregator = get_metrics(cfg)

    #init optimizer/scheduler/scaler
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg.warmup * steps_per_epoch,
                num_training_steps=cfg.epochs * steps_per_epoch,
                num_cycles=0.5
            )

    awp = None
    # if cfg.awp:
    #     awp = AWP(model, cfg.awp_start_epoch * steps_per_epoch, lr=cfg.awp_lr, eps=cfg.awp_eps)

    for epoch in range(cfg.epochs):
        gc.collect()
        model.train()
        torch.set_grad_enabled(True)
        if world_size > 1:
            train_ds.sampler.set_epoch(epoch)
        train_iterator = iter(train_ds)

        for k,v in metrics_aggregator.items():
            v.reset()

        for step in tqdm(range(steps_per_epoch), desc=f'Train epoch {epoch}', disable=rank!=0):
            global_step += 1

            data = next(train_iterator)

            result = step_fn(global_step, cfg, model, device, data, scaler, optimizer, scheduler, metrics_aggregator, rank, world_size, awp=awp, mode='train')

            if rank==0:
                if neptune_run:
                    neptune_log_const(neptune_run, "lr", value=optimizer.param_groups[0]["lr"], step=global_step)
                    if "grad_norm" in result.keys():
                        neptune_log_const(neptune_run, "grad_norm", value=result["grad_norm"], step=global_step)
                    for k,v in result["losses"].items():
                        neptune_log_const(neptune_run, k, value=v, step=global_step)

        for k,v in metrics_aggregator.items():
            value = v.compute(device=device, rank=rank, world_size=world_size)
            if rank == 0:
                if neptune_run:
                    neptune_log_const(neptune_run, f"train/{k}", value=value, step=global_step)
                print(f' - val_{k}: {value:.4f}', end='')
        if rank==0:
            print()

        if num_valid>0: #validation cond
            gc.collect()
            model.eval()
            torch.set_grad_enabled(False)
            valid_iterator = iter(valid_ds)

            for k,v in metrics_aggregator.items():
                v.reset()

            for step in tqdm(range(valid_steps_per_epoch), desc=f'Val epoch {epoch}', disable=rank!=0):
                data = next(valid_iterator)

                result = step_fn(global_step, cfg, model, device, data, scaler, optimizer, scheduler, metrics_aggregator, rank, world_size, awp=None, mode='valid')

            for k,v in metrics_aggregator.items():
                value = v.compute(device=device, rank=rank, world_size=world_size)
                if rank == 0:
                    if neptune_run:
                        neptune_log_const(neptune_run, f"valid/{k}", value=value, step=global_step)
                    print(f' - {k}: {value:.4f}', end='')
            if rank==0:
                print()
                
        if rank==0:
            torch.save({"model": model.state_dict(), "epoch": epoch, "optimizer":optimizer.state_dict(), "scheduler":scheduler.state_dict()}, f'{cfg.output_dir}/{cfg.model_path}')

    if rank==0:
        if neptune_run:
            neptune_run.stop()
    del train_ds, valid_ds
    gc.collect()
    
    # if num_valid>0 and rank == 0: #to check masking / metric
    #     model.eval()
    #     torch.set_grad_enabled(False)
    #     eval_ds = get_dataset(cfg, valid_files, mode='valid', batch_size=1, shuffle=False)
    #     eval_iterator = iter(eval_ds)
    #     mae_m = MAE_M()
    #     mae_m.reset()
    
    #     for step in tqdm(range(num_valid), desc=f'Eval epoch {epoch}'):
    #         data = next(eval_iterator)
    
    #         result = step_fn(global_step, cfg, model, data, scaler, optimizer, scheduler, metrics_aggregator, awp=None, mode='val')
    
    #         mae_m.update(result['output'], data['react'], device=device, rank=rank, world_size=world_size)
    
    #     print(f'eval_mae: {mae_m.compute()}')
    #     del eval_ds, eval_iterator, mae_m
    #     gc.collect()
    if world_size > 1:
        dist.destroy_process_group()
        
    return model


def train_fold(cfg, df, device, rank=0, world_size=1):
    if cfg.use_all_train:
        train = df
        valid = df[:0]
    else:
        df['fold'] = -1
        df = df.reset_index(drop=True)
        # df_2A3 = df.loc[df.experiment_type=='2A3_MaP'].reset_index(drop=True)
        # df_DMS = df.loc[df.experiment_type=='DMS_MaP'].reset_index(drop=True)
    
        kfold = KFold(n_splits=cfg.n_splits, random_state=cfg.seed, shuffle=True)
        for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(df)):
            # df_2A3.loc[valid_idx,'fold'] = fold_idx
            # df_DMS.loc[valid_idx,'fold'] = fold_idx
            df.loc[valid_idx, 'fold'] = fold_idx
            # print(f'fold{fold_idx}:', 'train', len(train_idx), 'valid', len(valid_idx))
    
        # df = pd.concat([df_2A3, df_DMS]).reset_index(drop=True)
    
        assert not (df['fold']==-1).sum()
        assert len(np.unique(df['fold']))==cfg.n_splits
    
        train = df.loc[df['fold']!=cfg.fold]
        valid = df.loc[df['fold']==cfg.fold]
    
    train_loop(cfg, train, valid, device, rank, world_size)