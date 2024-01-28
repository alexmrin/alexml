import torch.optim as optim

class LinearLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, total_steps: int, init_lr: float, end_lr: float, last_epoch: int = -1):
        self.total_steps = total_steps
        self.init_lr = init_lr
        self.end_lr = end_lr
        self.last_epoch = last_epoch
        pass

    def get_lr(self) -> float:
        return super().get_lr()