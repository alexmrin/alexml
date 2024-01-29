import torch

class Accuracy():
    def __init__(self):
        self.correct = 0
        self.total = 0

    def step(self, preds, labels):
        preds = torch.argmax(preds, dim=-1)
        self.correct += (preds == labels).sum().item()
        self.totol += labels.shape[0]

    def compute(self):
        if self.total == 0:
            return 0
        return self.correct / self.total
    
    def reset(self):
        self.correct = 0
        self.total = 0

class Recall():
    def __init__(self, mode: str = "macro"):
        self.total = 0
        self.true_positive = 0
        self.false_negative = 0
        if mode != ("macro" or "micro"):
            raise ValueError("Invalid mode. Pick between 'micro' and 'macro'.")
        self.mode = mode

    def step(self, preds, labels):
        pass

    def compute(self):
        pass

    def reset(self):
        pass

class Precision():
    def __init__(self):
        pass

    def step(self):
        pass

    def compute(self):
        pass

    def reset(self):
        pass

class F1Score():
    def __init__(self):
        pass

    def step(self):
        pass

    def compute(self):
        pass

    def reset(self):
        pass