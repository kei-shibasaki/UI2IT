import math 
import torch 


class CosineLRWarmup(torch.optim.lr_scheduler._LRScheduler):
    """
    optimizer: torch.optim.Optimizer, optimizer.
    lr_w: float, Learning rate when the training starts.
    lr_max: float, Learning rate when the warmup ends.
    lr_min: float, Learning rate when the training ends.
    step_w: int, Number of warm up steps.
    step_max: int, Number of training steps.
    """
    def __init__(self, optimizer, lr_w, lr_max, lr_min, step_w, step_max, last_epoch=-1, verbose=False):
        self.lr_w = lr_w 
        self.lr_max = lr_max 
        self.lr_min = lr_min 
        self.step_w = step_w 
        self.step_max = step_max
        super(CosineLRWarmup, self).__init__(optimizer, last_epoch, verbose)
    
    def func1(self, step):
        return (self.lr_max-self.lr_w)/self.step_w * step + self.lr_w
    
    def func2(self, step):
        return 0.5*(self.lr_max-self.lr_min)*math.cos(math.pi*(step-self.step_w)/(self.step_max-self.step_w)) + 0.5*(self.lr_max+self.lr_min)
    
    def get_lr(self):
        step = self.last_epoch + 1
        func = self.func1 if step<self.step_w else self.func2
        return [func(step) for base_lr in self.base_lrs]