import torch

class AWP():
  def __init__(self,
               model,
               start_step=0,
               param_name="weight",
               lr=0.1,
               eps=0.):
    self.model = model
    self.param_name = param_name
    self.lr = 1 if lr is None else lr
    self.eps = 0.001 if eps is None else eps
    self.start_step = start_step
    self.backup = {}
    self.backup_eps = {}

  def will_skip(self, step):
    return (self.lr == 0) or (step < self.start_step)

  def on_retrain_begin(self, step=0):
    if self.will_skip(step):
      return False

    self.save()
    self.attack()
    return True

  def on_retrain_end(self, step=0):
    if self.will_skip(step):
      return False

    self.restore()
    return True

  def attack(self):
    e = 1e-6
    for name, param in self.model.named_parameters():
      if param.requires_grad and param.grad is not None and self.param_name in name:
        norm1 = torch.norm(param.grad)
        norm2 = torch.norm(param.data.detach())
        if norm1 != 0 and not torch.isnan(norm1):
          r_at = self.lr * param.grad / (norm1 + e) * (norm2 + e)
          param.data.add_(r_at)
          param.data = torch.min(
              torch.max(param.data, self.backup_eps[name][0]),
              self.backup_eps[name][1])
        # param.data.clamp_(*self.backup_eps[name])

  def save(self):
    for name, param in self.model.named_parameters():
      if param.requires_grad and param.grad is not None and self.param_name in name:
        if name not in self.backup:
          self.backup[name] = param.data.clone()
          grad_eps = self.eps * param.abs().detach()
          self.backup_eps[name] = (
              self.backup[name] - grad_eps,
              self.backup[name] + grad_eps,
          )

  def restore(self,):
    for name, param in self.model.named_parameters():
      if name in self.backup:
        param.data = self.backup[name]
    self.backup = {}
    self.backup_eps = {}