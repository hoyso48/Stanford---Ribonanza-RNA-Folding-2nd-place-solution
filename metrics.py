import torch

def my_gather(output, rank, world_size):
    # output must be a tensor on the cuda device(nccl)
    # output must have the same size in all workers
    result = None
    if rank == 0:
        result = [torch.empty_like(output) for _ in range(world_size)]
    torch.distributed.gather(output, gather_list=result, dst=0)
    return result

class AverageMeter(object):
    """
       Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def compute(self, device=None, rank=0, world_size=1):
        if world_size > 1:
            sum = my_gather(self.sum, rank, world_size)
            count = my_gather(self.count, rank, world_size)
            if rank == 0:
                self.sum = torch.stack(sum).sum()
                self.count = torch.stack(count).sum()
        return (self.sum / self.count).item()

class MAE(AverageMeter):
    def update(self, result):
        super().update(result['metrics']['mae'], result['react_sample_size'])
        
def get_metrics(cfg):
    return {'mae': MAE()}


