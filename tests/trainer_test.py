import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

alexml_path = os.path.join("..", "alexml")
sys.path.append(alexml_path)

from alexml.training import Trainer, TrainingArgs

# Define a simple linear model
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 30) 
        self.linear2 = nn.Linear(30, 10)
        self.bn = nn.BatchNorm1d(30)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.linear1(x)
        return self.linear2(self.bn(F.relu(x)))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
eval_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# Set up training arguments
args = TrainingArgs(
    lr=0.0005,
    num_epochs=3,
    batch_size=128,
    save_steps=500,
)

model = TestModel()

# Instantiate the Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    optimizer=optim.AdamW,
    criterion=nn.CrossEntropyLoss
)

if __name__ == "__main__":
    trainer.train()
    trainer.plot()