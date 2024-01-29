import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from alexml.utils import check_type, exponential_moving_average

class TrainingArgs():
    """
    Holds training configuration for a PyTorch model training session.

    Attributes:
        lr (float): Learning rate for the optimizer.
        adam_beta1 (float): Beta1 hyperparameter for Adam optimizer.
        adam_beta2 (float): Beta2 hyperparameter for Adam optimizer.
        warmup_steps (int): Initial steps with a lower learning rate.
        warmup_ratio (Optional[float]): Ratio for gradually increasing the learning rate.
        num_epochs (int): Total number of training epochs.
        weight_decay (float): L2 penalty term for regularization.
        device (Optional[str]): Computation device ('cuda' or 'cpu'), auto-selected if None.
        batch_size (int): Number of samples per training batch.
        save_steps (Optional[int]): Step frequency for saving model checkpoints.
        save_dir (str): Directory path for saving model checkpoints.
        num_workers (int): Number of subprocesses for data loading.
        evaluation_strategy (str): Strategy ('epoch' or 'batch') for model evaluation.

    Methods:
        __init__: Initializes TrainingArgs with specified configuration.
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
        evaluation_strategy = "epoch"
    ):
        check_type(lr, float)
        check_type(adam_beta1, float)
        check_type(adam_beta2, float)
        check_type(warmup_steps, int)
        check_type(warmup_ratio, float, allow_none=True)
        check_type(num_epochs, int)
        check_type(weight_decay, float)
        check_type(device, str, allow_none=True) # fix later to check for supported strings
        check_type(batch_size, int)
        check_type(save_steps, int, allow_none=True) # fix later to check its a valid filepath?
        check_type(save_dir, str)
        check_type(num_workers, int)
        check_type(evaluation_strategy, str) # check if its epoch or batch

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
        self.evaluation_strategy = evaluation_strategy

class Trainer():
    """
    Manages the training and evaluation process of a PyTorch model.

    Attributes:
        model (nn.Module): The neural network model to be trained.
        args (TrainingArgs): Configuration parameters for training.
        train_dataset (Dataset): Dataset used for training the model.
        eval_dataset (Optional[Dataset]): Dataset used for model evaluation, if provided.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        criterion (nn.modules.loss._Loss): Loss function used during training.
        lr_scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        eval_dataloader (Optional[DataLoader]): DataLoader for the evaluation dataset, if provided.
        train_losses (list): List of training losses for each batch.
        eval_losses (list): List of evaluation losses for each step or epoch.
        current_epoch (int): Current training epoch.
        current_step (int): Current step within the current epoch.

    Methods:
        __init__: Initializes the Trainer with model, datasets, and training arguments.
        plot: Plots training and evaluation losses over time.
        train: Executes the training loop over the specified number of epochs.
        _train_epoch: Performs training operations for a single epoch.
        _train_batch: Processes a single batch for training and returns the loss.
        _evaluate: Evaluates the model on the evaluation dataset.
        _save_checkpoint: Saves the model state and training progress.
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
        check_type(eval_dataset, Dataset, allow_none=True)
        check_type(optimizer, torch.optim.Optimizer)
        check_type(criterion, nn.modules.loss._Loss)
        check_type(lr_scheduler, torch.optim.lr_scheduler._LRScheduler, allow_none=True)

        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.num_epochs = args.num_epochs
        self.optimizer = optimizer(self.model.parameters(), lr=self.args.lr, betas=(self.args.adam_beta1, self.args.adam_beta2), weight_decay=self.args.weight_decay)
        self.criterion = criterion()
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        if eval_dataset is not None:
            self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        self.scheduler = lr_scheduler(self.optimizer, max_lr=self.args.lr, total_steps=self.num_epochs*len(self.train_dataloader))
        self.train_losses = []
        self.eval_losses = []
        self.current_epoch = 0
        self.current_step = 1

    def plot(self) -> None:
        """
        Generates a plot of the training and evaluation losses.

        This method visualizes the training and evaluation losses over time, using an
        exponential moving average to smooth the loss values for better readability. 
        It produces a side-by-side plot with two subplots: one for training losses and 
        another for evaluation losses.

        The x-axis represents the number of batches or evaluation steps, and the y-axis 
        represents the smoothed loss values. This visualization aids in understanding the 
        model's learning trend and identifying issues like overfitting or underfitting.

        Returns:
            None: This method displays the plot but does not return any value.
        """
        fig, axs = plt.subplots(1, 2, figsize=(9, 4))
        train_y = np.array(self.train_losses)
        train_y = exponential_moving_average(train_y, 0.1)
        train_x = np.arange(len(self.train_losses))
        eval_y = np.array(self.eval_losses)
        eval_y = exponential_moving_average(eval_y, 0.1)
        eval_x = np.arange(len(self.eval_losses))
        axs[0].plot(train_x, train_y)
        axs[0].set_title("Train Losses")
        axs[1].plot(eval_x, eval_y)
        axs[1].set_title("Eval Losses")
        plt.show()

    def train(self) -> None:
        """
        Executes the training loop over the specified number of epochs.

        This method handles the complete training process for the model. It iterates 
        through each epoch, performs training on the training dataset, and conducts 
        evaluations based on the specified evaluation strategy. It also ensures that 
        the model and data are on the correct device (CPU or GPU).

        After completing the training for all epochs, it saves the final model state.

        Returns:
            None
        """
        self.model.to(self.args.device)
        for epoch in range(self.current_epoch, self.num_epochs):
            self._train_epoch(epoch)
            if self.args.evaluation_strategy == "epoch":
                self._evaluate(epoch)
            self.current_epoch += 1
        self._save_checkpoint(final=True)

    def _train_epoch(self, epoch: int) -> None:
        """
        Trains the model for one epoch.

        Iterates over the training dataset in batches and performs the training 
        operations for each batch. It accumulates the loss over all batches to 
        compute the average loss for the epoch.

        Args:
            epoch (int): The current epoch number in the training process.

        Returns:
            None
        """
        self.model.train()
        total_loss = 0
        total_samples = 0
        for inputs, labels in tqdm(self.train_dataloader):
            loss, num_samples = self._train_batch(inputs, labels)
            total_loss += loss
            total_samples += num_samples
        total_loss /= total_samples
        print(f"Epoch: {epoch} train loss: {total_loss}")

    def _train_batch(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Processes and trains a single batch of data.

        This method performs the forward pass, computes the loss, and executes backpropagation 
        and optimization steps for a single batch. It handles both training operations and, 
        optionally, learning rate scheduling and batch-based evaluation.

        The method ensures all computations are performed on the appropriate device (CPU or GPU). 
        It also tracks the training progress, appending the loss of each batch to a list for later analysis.

        Args:
            inputs (Tensor): The input data for the batch. It should be of the shape expected by the model.
            labels (Tensor): The corresponding labels for the input data.

        Returns:
            tuple: A tuple containing:
                - float: The loss value for the batch.
                - int: The number of samples in the batch.
        """
        inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
        self.optimizer.zero_grad()
        preds = self.model(inputs)
        loss = self.criterion(preds, labels)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.args.evaluation_strategy == "batch":
            self._evaluate(batch=True)
        self.current_step += 1
        self.train_losses.append(loss.item())
        if self.current_step % self.args.save_steps == 0:
            self._save_checkpoint()
        return loss.item(), inputs.shape[0]

    def _evaluate(self, epoch: int, batch: bool = False) -> None:
        """
        Evaluates the model on the evaluation dataset.

        Switches the model to evaluation mode and iterates over the evaluation dataset
        to compute the total loss. It keeps track of evaluation loss for each step or 
        epoch based on the provided arguments.

        Args:
            epoch (int): The current epoch number.
            batch (bool, optional): Indicates whether the evaluation is per batch 
                                    or per epoch. Default is True.

        Returns:
            None
        """
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_samples = 0
            for inputs, labels in tqdm(self.eval_dataloader):
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                preds = self.model(inputs)
                loss = self.criterion(preds, labels)
                total_loss += loss.item()
                self.eval_losses.append(loss.item())
                total_samples += inputs.shape[0]
            total_loss /= total_samples
            if batch:
                print(f"step {self.current_step} eval loss: {total_loss}")
            else:
                print(f"Epoch {epoch} eval loss: {total_loss}")
    
    def _save_checkpoint(self, final: bool = False) -> None:
        """
        Saves the current state of the model and training process.

        Creates a checkpoint with the model state, optimizer state, scheduler state, 
        and loss records. The checkpoint can be used for resuming training or 
        model evaluation. Optionally saves the final model state at the end of training.

        Args:
            final (bool, optional): Indicates if this is the final model save 
                                    at the end of training. Default is False.

        Returns:
            None
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.current_epoch,
            "step": self.current_step,
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses if self.eval_dataset is not None else []
        }
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        filename = f"{self.args.save_dir}/checkpoint-epoch-{self.current_epoch}-step-{self.current_step}.pth" if not final else f"{self.args.save_dir}/trained_model.pth"
        torch.save(checkpoint, filename)