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

class LinearLRWarmup(torch.optim.lr_scheduler._LRScheduler):
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
        super(LinearLRWarmup, self).__init__(optimizer, last_epoch, verbose)
    
    def func1(self, step):
        return (self.lr_max-self.lr_w)/self.step_w * step + self.lr_w
    
    def func2(self, step):
        alpha = (self.lr_min-self.lr_max)/(self.step_max-self.step_w)
        beta = self.lr_max - alpha*self.step_w
        return alpha*step + beta
    
    def get_lr(self):
        step = self.last_epoch + 1
        func = self.func1 if step<self.step_w else self.func2
        return [func(step) for base_lr in self.base_lrs]

class LinearLRWarmup2(torch.optim.lr_scheduler._LRScheduler):
    """
    optimizer: torch.optim.Optimizer, optimizer.
    lr_w: float, Learning rate when the training starts.
    lr_max: float, Learning rate when the warmup ends.
    lr_min: float, Learning rate when the training ends.
    step_0: int, Number of warm up steps.
    step_2: int, Number of training steps.
    """
    def __init__(self, optimizer, lr_w, lr_max, lr_min, step_0, step_1, step_2, last_epoch=-1, verbose=False):
        self.lr_w = lr_w 
        self.lr_max = lr_max 
        self.lr_min = lr_min 
        self.step_0 = step_0 
        self.step_1 = step_1
        self.step_2 = step_2
        super(LinearLRWarmup2, self).__init__(optimizer, last_epoch, verbose)
    
    def func0(self, step):
        return (self.lr_max-self.lr_w)/self.step_0 * step + self.lr_w
    
    def func1(self, step):
        return self.lr_max
    
    def func2(self, step):
        alpha = (self.lr_min-self.lr_max)/(self.step_2-self.step_1)
        beta = self.lr_max - alpha*self.step_1
        return alpha*step + beta
    
    def get_func(self, step):
        if step<self.step_0:
            return self.func0
        elif step<self.step_1:
            return self.func1
        else:
            return self.func2
    
    def get_lr(self):
        step = self.last_epoch + 1
        func = self.get_func(step)

        return [max(func(step), 0.0) for base_lr in self.base_lrs]