import pandas as pd
import numpy as np
import torch
from data import get_dataset
from model import RNARegModel
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

def get_predictions(model, df, cfg):
    ds = get_dataset(cfg, df, mode='test', shuffle=False, batch_size=cfg.batch_size, drop_last=False, dir=cfg.data_dir)
    device = cfg.device
    model = model.to(device)
    model.eval()
    _iterator = iter(ds)
    predictions = []
    with torch.no_grad():
        for step in tqdm(range(len(ds)), desc=f'predicting'):
            data = next(_iterator)
            data = dict([(k, v.to(device)) for k, v in data.items()])
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                result = model(data)
            predictions.append(result['output'])
    return predictions

def predictions_to_sub(predictions, df):
    preds = [x.detach().cpu().float().numpy() for x in predictions]
    preds = [item for sublist in preds for item in sublist]
    preds = np.concatenate([x[:l] for x,l in zip(preds, df.seq_len.values)])
    assert len(preds) == 269796671
    sub_df = pd.DataFrame({'id':np.arange(0, 269796671, 1), 
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
    p_DMS = np.zeros((269796671,), dtype=np.float32)
    p_2A3 = np.zeros((269796671,), dtype=np.float32)
    for p, w in zip(sub_dfs, weights):
        df = pd.read_parquet(p)
        if plot:
            plt.plot(df.iloc[-457:-150,1:3])
            plt.show()
        
        p_DMS += df['reactivity_DMS_MaP'].values * w
        p_2A3 += df['reactivity_2A3_MaP'].values * w
        del df
    if clip:
        p_DMS = np.clip(p_DMS, 0, 1)
        p_2A3 = np.clip(p_2A3, 0, 1)
    sub_df = pd.DataFrame({'id':np.arange(0, 269796671, 1), 
                               'reactivity_DMS_MaP':p_DMS, 
                               'reactivity_2A3_MaP':p_2A3})
    if plot:
        plt.plot(sub_df.iloc[-457:-150,1:3])
        plt.show()
    return sub_df

def make_submission(cfg, model_paths, df, filename='submission.parquet', weights=None, clip=True):
    fns = []
    for p in model_paths:
        loaded = False
        model = RNARegModel(cfg.dim, cfg.num_layers, cfg.num_heads)
        loaded_state_dict = torch.load(f'{p}')['model']
        try:
            model.load_state_dict(loaded_state_dict, strict=True)
            loaded = True
        except:
            pass
            
        model = torch.compile(model, dynamic=True)
    
        if not loaded:
                model.load_state_dict(loaded_state_dict, strict=True)
            
        model.eval()
        predictions = get_predictions(model, df, cfg)
        sub = predictions_to_sub(predictions)
        fn = f'{p[:-4]}-sub.parquet'
        print(f'wrote {fn}')
        sub.to_parquet(fn)
        fns.append(fn)
        del sub, predictions, model
        gc.collect()
    sub = merge_submissions(fns, clip=True, weights=weights, plot=True)
    sub.to_parquet(filename)
    return sub
    
        
    