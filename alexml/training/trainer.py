from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import check_type

class TrainingArgs():
    """
    A class to hold the training arguments for a PyTorch training session.

    Attributes:
        lr (float): Learning rate for the optimizer.
        adam_beta1 (float): Beta1 parameter for Adam optimizer.
        adam_beta2 (float): Beta2 parameter for Adam optimizer.
        warmup_steps (int): Number of warmup steps for learning rate scheduling.
        warmup_ratio (Optional[float]): Warmup ratio for learning rate scheduling.
        num_epochs (int): Number of epochs to train the model.
        weight_decay (float): Weight decay for regularization.
        device (str): Device to run the training on ('cuda' or 'cpu').
        batch_size (int): Batch size for training.
        save_steps (Optional[int]): Frequency of saving the model.
        save_dir (str): Directory to save the model.
        num_workers (int): Number of workers for data loading.

    Methods:
        __init__: Constructs all the necessary attributes for the TrainingArgs object.
    """
    def __init__(
        self,
        lr: float = 5e-5,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        warmup_steps: int = 0,
        warmup_ratio: Optional[float] = None,
        num_epochs: int = None,
        weight_decay: float = 0.01,
        device: Optional[str] = None,
        batch_size: int = None,
        save_steps: Optional[int] = None,
        save_dir: str = "./saves",
        num_workers: int = 1,
    ):
        check_type(lr, float)
        check_type(adam_beta1, float)
        check_type(adam_beta2, float)
        check_type(warmup_steps, int)
        check_type(warmup_ratio, float, allow_none=True)
        check_type(num_epochs, int)
        check_type(weight_decay, float)
        check_type(device, str, allow_none=True)
        check_type(batch_size, int)
        self.lr = lr
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.batch_size = batch_size
        self.save_steps = save_steps
        self.save_dir = save_dir
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = warmup_steps
        self.num_workers = num_workers

class Trainer():
    """
    A class to manage the training process of a PyTorch model.

    Attributes:
        model (nn.Module): The model to be trained.
        args (TrainingArgs): The training arguments.
        train_dataset (Dataset): The dataset for training the model.
        eval_dataset (Optional[Dataset]): The dataset for evaluating the model.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        criterion (nn.modules.loss._Loss): The loss function.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        train_dataloader (DataLoader): DataLoader for training data.
        eval_dataloader (Optional[DataLoader]): DataLoader for evaluation data.
        
    Methods:
        __init__: Constructs all the necessary attributes for the Trainer object.
        train: Runs the training loop over the specified number of epochs.
    """
    def __init__(
        self,
        model: nn.Module = None,
        args: TrainingArgs = None,
        train_dataset: Dataset = None,
        eval_dataset: Optional[Dataset]= None,
        optimizer: torch.optim.Optimizer = torch.optim.AdamW,
        criterion: nn.modules.loss._Loss = nn.CrossEntropyLoss,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = torch.optim.lr_scheduler.OneCycleLR
):
        check_type(model, nn.Module)
        check_type(args, TrainingArgs)
        check_type(train_dataset, Dataset)
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.num_epochs = args.num_epochs
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr, betas=(self.args.adam_beta1, self.args.adam_beta2), weight_decay=self.args.weight_decay)
        self.criterion = criterion()
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        if eval_dataset is not None:
            self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        self.lr_scheduler = lr_scheduler(self.optimizer, max_lr=self.args.lr, total_steps=self.num_epochs*len(self.train_dataloader))


    def train(self) -> None:
        """
        Runs the training loop over the specified number of epochs.

        This method iterates over the training dataset, computes the loss,
        and updates the model parameters. Additionally, it evaluates the model
        on the evaluation dataset if provided.

        Returns:
            None
        """
        train_losses = []
        eval_losses = []
        step = 0
        self.model.to(self.args.device)
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            for inputs, labels in tqdm(self.train_dataloader):
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                step += 1
                self.optimizer.zero_grad()
                preds = self.model(inputs)
                loss = self.criterion(preds, labels)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                train_losses.append(loss.item())
            epoch_loss /= len(self.train_dataloader)
            print(f"Epoch {epoch} train loss: {epoch_loss}")
            if self.eval_dataset is not None:
                self.model.eval()
                with torch.no_grad():
                    epoch_loss = 0
                    for inputs, labels in tqdm(self.eval_dataloader):
                        inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                        preds = self.model(inputs)
                        loss = self.criterion(preds, labels)
                        epoch_loss += loss.item()
                        eval_losses.append(loss.item())
                    epoch_loss /= len(self.eval_dataloader)
                    print(f"Epoch {epoch} eval loss: {epoch_loss}")
