import torch
import numpy as np
import random


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


def calculate_accuracy(pred:torch.Tensor, batch_y:torch.Tensor):
    predicted_classes = torch.argmax(pred, dim=1)
    correct = (predicted_classes == batch_y).float().sum()
    accuracy = (correct/batch_y.shape[0]).item()
    return accuracy

def reproducible(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    