import torch
import torch.nn.functional as F
from fastai.vision.all import Metric, to_detach, num_distrib, maybe_gather

class MAE(Metric):
    def __init__(self): 
        self.reset()
        
    def reset(self): 
        self.x,self.y = [],[]
        
    def accumulate(self, learn):
        x = learn.pred['react_pred'][learn.y['mask']]
        y = learn.y['react'][learn.y['mask']].clip(0,1)
        #this implementation cannot support DDP since sizes of x,y can be vary on different devices
        self.x.append(x)
        self.y.append(y)

    @property
    def value(self):
        x,y = torch.cat(self.x,0),torch.cat(self.y,0)
        loss = F.l1_loss(x, y, reduction='none')
        loss = loss[~torch.isnan(loss)].mean()
        return loss

class MAE_ave(Metric):
    '''
    memory-efficient version of the MAE metric with DDP support,
    equivalent in value to class MAE.
    '''
    def __init__(self): 
        self.reset()
        
    def reset(self): 
        self.sum, self.n = 0, 0
        
    def accumulate(self, learn):
        x = learn.pred['react_pred'][learn.y['mask']]
        y = learn.y['react'][learn.y['mask']].clip(0,1)
        
        mae = F.l1_loss(x, y, reduction='none')
        msk = ~torch.isnan(mae)
        sum_ = mae[msk].double().sum() / 1000.
        n_ = msk.double().sum() / 1000.

        sum_, n_ = to_detach(sum_), to_detach(n_)

        self.sum += sum_
        self.n += n_
        
    @property
    def value(self):
        return self.sum / self.n


