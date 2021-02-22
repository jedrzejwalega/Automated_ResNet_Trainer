import torch
import numpy as np


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_cuda_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    c = torch.cuda.memory_cached(0)
    a = torch.cuda.memory_allocated(0)
    f = c-a  # free inside cache
    print(f"Total: {t}\nCached: {c}\nAllocated: {a}\nFree: {f}\n\n")

def check_for_nan(self, losses, nan_replacement):
    losses = [loss if not np.isnan(loss) else nan_replacement for loss in losses]
    return losses