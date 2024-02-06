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
        self.convs = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.SiLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
eval_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)

# Set up training arguments
args = TrainingArgs(
    lr=0.001,
    num_epochs=10,
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
    criterion=nn.CrossEntropyLoss,
    metrics=["accuracy", "recall", "precision", "f1"]
)

if __name__ == "__main__":
    trainer.train()
    trainer.plot()