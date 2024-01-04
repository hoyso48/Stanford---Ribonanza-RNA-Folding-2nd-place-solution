from fastai.vision.all import TrackerCallback, rank_distrib
import torch

# %% ../nbs/00_torch_core.ipynb 170
def get_model2(model):
    "Return the model maybe wrapped inside `model`."
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        return model.module
    elif hasattr(model, '_orig_mod'):
        return model._orig_mod #for torch.compile model
    else:
        return model

class SaveLastCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model at the end of every epoch"
    order = TrackerCallback.order+1
    def __init__(self, 
        fname='model', # model name to be used when saving model.
        with_opt=True, # if true, save optimizer state (if any available) when saving model.
        with_epoch=True, # if true, save epoch
    ):
        super().__init__()
        self.fname = fname
        self.with_opt = with_opt
        self.with_epoch = with_epoch

    def _save(self, name):
        if rank_distrib(): return
        else:
            model = get_model2(self.model).state_dict()
            state = {'model':model}
            opt = self.opt
            if self.with_opt and opt is not None:
                state['opt'] = opt.state_dict()
            if self.with_epoch:
                state['epoch'] = self.epoch
                
            torch.save(state, f'{self.path}/{self.model_dir}/{name}.pth')

    def after_epoch(self):
        self._save(f'{self.fname}')


