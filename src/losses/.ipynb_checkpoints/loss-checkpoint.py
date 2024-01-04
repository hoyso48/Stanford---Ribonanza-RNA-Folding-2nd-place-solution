import torch
import torch.nn.functional as F

def mae_loss(pred,target):
    p = pred['react_pred'][target['mask']]
    y = target['react'][target['mask']].clip(0,1)
    loss = F.l1_loss(p, y, reduction='none')
    loss = loss[~torch.isnan(loss)].mean()
    
    return loss
    
def weighted_mae_loss(pred,target):
    p = pred['react_pred']
    y = target['react'].clip(0,1)
    mask = target['mask']
    weights = torch.log1p(target['snr'].clip(0)).clip(0,10).unsqueeze(1)
    loss = F.l1_loss(p, y, reduction='none')
    loss = loss * weights
    loss = loss[mask]
    loss = loss[~torch.isnan(loss)].mean()
    
    return loss