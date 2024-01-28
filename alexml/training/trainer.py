from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from alexml.utils import check_type

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
        evaluation_strategy (str): Evaluate every 'epoch' or every 'batch'.

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
    Manages the training process of a PyTorch model.

    Attributes:
        model (nn.Module): The neural network model to train.
        args (TrainingArgs): Configuration for training parameters.
        train_dataset (Dataset): Dataset for training the model.
        eval_dataset (Optional[Dataset]): Dataset for evaluating the model, if any.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (nn.modules.loss._Loss): Loss function for training.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler for learning rate.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        eval_dataloader (Optional[DataLoader]): DataLoader for the evaluation dataset.
        train_losses (list): List to record training loss per batch.
        eval_losses (list): List to record evaluation loss per batch/epoch.
        current_epoch (int): Tracker for the current epoch.
        current_step (int): Tracker for the current step within an epoch.

    Methods:
        __init__: Initializes the Trainer with the model, datasets, and training arguments.
        train: Executes the training loop over the specified number of epochs.
        _train_epoch: Trains the model for one epoch.
        _train_batch: Trains the model for one batch and returns the loss.
        _evaluate: Evaluates the model on the evaluation dataset.
        _save_checkpoint: Saves the model state, along with training and evaluation losses.
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
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr, betas=(self.args.adam_beta1, self.args.adam_beta2), weight_decay=self.args.weight_decay)
        self.criterion = criterion()
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        if eval_dataset is not None:
            self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        self.scheduler = lr_scheduler(self.optimizer, max_lr=self.args.lr, total_steps=self.num_epochs*len(self.train_dataloader))
        self.train_losses = []
        self.eval_losses = []
        self.current_epoch = 0
        self.current_step = 1

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
        self._save_checkpoint(self.current_epoch, final=True)

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
        for inputs, labels in tqdm(self.train_dataloader):
            total_loss += self._train_batch(inputs, labels)
        total_loss /= len(self.train_dataloader)
        print(f"Epoch: {epoch} train loss: {total_loss}")

    def _train_batch(self, inputs, labels) -> float:
        """
        Executes training operations for a single batch.

        Processes the inputs and labels, computes the loss, and performs backpropagation 
        and optimization steps. Optionally, it also steps the learning rate scheduler and 
        evaluates the model if the evaluation strategy is set to 'batch'.

        Args:
            inputs: Input data for the batch.
            labels: Corresponding labels for the batch.

        Returns:
            float: The loss value for the batch.
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
        return loss.item()

    def _evaluate(self, epoch: int, batch: bool = True) -> None:
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
            for inputs, labels in tqdm(self.eval_dataloader):
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                preds = self.model(inputs)
                loss = self.criterion(preds, labels)
                total_loss += loss.item()
                self.eval_losses.append(loss.item())
            total_loss /= len(self.eval_dataloader)
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
        filename = f"{self.args.save_dir}/checkpoint-epoch-{self.current_epoch}-step-{self.current_step}" if not final else f"{self.args.save_dir}/trained_model"
        torch.save(checkpoint, filename)