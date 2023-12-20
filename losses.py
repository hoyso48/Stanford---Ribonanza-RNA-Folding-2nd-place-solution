import torch
import torch.nn.functional as F

def mae_loss(pred,target,mask,weights=None):
    p = pred
    y = target.clip(0,1)
    loss = F.l1_loss(p, y, reduction='none')
    if weights is not None:
        loss = loss * weights
    loss = loss[mask]
    msk = ~torch.isnan(loss)
    sample_size = msk.long().sum()
    loss = loss[msk].mean()
    return loss, sample_size

# class MAE(object):
#     def __init__(self): 
#         self.reset()
        
#     def reset(self): 
#         self.x,self.y = [],[]
        
#     def accumulate(self, pred, y):
#         assert pred.size(0) == 1
#         assert y.size(0) == 1
#         x = pred[0]
#         y = y[0].clip(0,1)
#         self.x.append(x)
#         self.y.append(y)

#     def value(self):
#         x,y = torch.cat(self.x,0),torch.cat(self.y,0)
#         loss = F.l1_loss(x, y, reduction='none')
#         loss = loss[~torch.isnan(loss)].mean()
#         return loss
