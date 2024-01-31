import torch

class Accuracy():
    """
    A class to compute the accuracy metric in a machine learning context.

    Attributes:
        correct (int): The number of correctly predicted instances.
        total (int): The total number of instances evaluated.

    Methods:
        step(preds, labels): Updates the count of correct predictions and total instances.
        compute(): Calculates and returns the accuracy metric.
        reset(): Resets the count of correct predictions and total instances.
    """
    def __init__(self):
        self.reset()

    def step(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update the metric with a new batch of predictions and labels.

        Args:
            preds (torch.Tensor): Predictions from the model, as logits or probabilities.
            labels (torch.Tensor): Ground truth labels.
        """
        preds = torch.argmax(preds, dim=-1)
        self.correct += (preds == labels).sum().item()
        self.total += labels.shape[0]

    def compute(self) -> float:
        """
        Compute and return the accuracy metric.

        Returns:
            float: The accuracy calculated as the ratio of correct predictions to total predictions.
        """
        return self.correct / (self.total + 1e-10)
    
    def reset(self) -> None:
        """
        Reset the counts of correct predictions and total instances.
        """
        self.correct = 0
        self.total = 0

class Recall():
    """
    A class to compute the recall metric in a machine learning context, supporting both micro and macro averaging.

    Attributes:
        mode (str): The averaging mode, either 'micro' or 'macro'.
        num_classes (int): The number of classes in the dataset.
        true_positive (torch.Tensor): Counts of true positives per class.
        false_negative (torch.Tensor): Counts of false negatives per class.
        initialized (bool): Indicates if the class has been initialized with the number of classes.

    Methods:
        step(preds, labels): Updates the count of true positives and false negatives for each class.
        compute(): Calculates and returns the recall metric.
        reset(): Resets the counts of true positives and false negatives, and the initialized state.
    """
    def __init__(self, mode: str = "micro"):
        if mode not in ["macro", "micro"]:
            raise ValueError("Invalid mode. Pick between 'micro' and 'macro'.")
        self.mode = mode
        self.num_classes = None
        self.reset()

    def step(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update the metric with a new batch of predictions and labels.

        Args:
            preds (torch.Tensor): Predictions from the model, as logits or probabilities.
            labels (torch.Tensor): Ground truth labels.
        """
        if not self.initialized:
            self.num_classes = preds.shape[-1]
            self.true_positive = torch.zeros(self.num_classes)
            self.false_negative = torch.zeros(self.num_classes)
            self.initialized = True

        preds = torch.argmax(preds, dim=-1)
        for cls in range(self.num_classes):
            mask = cls == labels
            self.true_positive[cls] += ((preds == cls) & mask).sum().item()
            self.false_negative[cls] += ((preds != cls) & mask).sum().item()

    def compute(self) -> float:
        """
        Compute and return the recall metric.

        Returns:
            float: The recall metric, calculated according to the specified mode.
        """
        if self.mode == "micro":
            tp_sum = self.true_positive.sum()
            fn_sum = self.false_negative.sum()
            return (tp_sum / (tp_sum + fn_sum + 1e-10)).item()
        else:
            total_recall = self.true_positive / (self.true_positive + self.false_negative + 1e-10)
            return total_recall.mean().item()

    def reset(self) -> None:
        """
        Reset the counts of true positives, false negatives, and the initialized state.
        """
        self.true_positive = None
        self.false_negative = None
        self.initialized = False

class Precision():
    """
    A class for calculating Precision in a multi-class classification setting.
    
    Attributes:
        mode (str): The mode for Precision calculation - 'micro' or 'macro'.
        num_classes (int): Number of classes in the classification task.
        true_positive (torch.Tensor): Tensor to keep track of true positives for each class.
        false_positive (torch.Tensor): Tensor to keep track of false positives for each class.
        initialized (bool): Flag to check if the class has been initialized.
    """
    def __init__(self, mode: str = "micro"):
        if mode not in ["macro", "micro"]:
            raise ValueError("Invalid mode. Pick between 'micro' and 'macro'.")
        self.mode = mode
        self.num_classes = None
        self.reset()

    def step(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update the state of true positives and false positives based on the predictions and labels.

        Args:
            preds (torch.Tensor): Predictions from the model. Shape: (batch_size, num_classes)
            labels (torch.Tensor): Ground truth labels. Shape: (batch_size,)
        """
        if not self.initialized:
            self.num_classes = preds.shape[-1]
            self.true_positive = torch.zeros(self.num_classes)
            self.false_positive = torch.zeros(self.num_classes)
            self.initialized = True

        preds = torch.argmax(preds, dim=-1)
        for cls in range(self.num_classes):
            mask = cls == labels
            self.true_positive[cls] += ((preds == cls) & mask).sum().item()
            self.false_positive[cls] += ((preds == cls) & ~mask).sum().item()

    def compute(self) -> float:
        """
        Compute the Precision based on the mode specified during initialization.

        Returns:
            float: The computed Precision.
        """
        if self.mode == "micro":
            tp_sum = self.true_positive.sum()
            fp_sum = self.false_positive.sum()
            return (tp_sum / (tp_sum + fp_sum + 1e-10)).item()
        else:
            total_precision = self.true_positive / (self.true_positive + self.false_positive + 1e-10)
            return total_precision.mean().item()

    def reset(self) -> None:
        """
        Reset the state of the metrics (true positives, false positives) to initial state.
        """
        self.true_positive = None
        self.false_positive = None
        self.initialized = False

class F1Score():
    """
    A class for calculating F1 Score in a multi-class classification setting.
    
    Attributes:
        mode (str): The mode for F1 calculation - 'micro' or 'macro'.
        num_classes (int): Number of classes in the classification task.
        true_positive (torch.Tensor): Tensor to keep track of true positives for each class.
        false_positive (torch.Tensor): Tensor to keep track of false positives for each class.
        false_negative (torch.Tensor): Tensor to keep track of false negatives for each class.
        initialized (bool): Flag to check if the class has been initialized.
    """

    def __init__(self, mode: str = "micro"):
        if mode not in ["macro", "micro"]:
            raise ValueError("Invalid mode. Pick between 'micro' and 'macro'.")
        self.mode = mode
        self.num_classes = None
        self.reset()

    def step(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update the state of true positives, false positives, and false negatives based on the predictions and labels.

        Args:
            preds (torch.Tensor): Predictions from the model. Shape: (batch_size, num_classes)
            labels (torch.Tensor): Ground truth labels. Shape: (batch_size,)
        """
        if not self.initialized:
            self.num_classes = preds.shape[-1]
            self.true_positive = torch.zeros(self.num_classes)
            self.false_negative = torch.zeros(self.num_classes)
            self.false_positive = torch.zeros(self.num_classes)
            self.initialized = True

        preds = torch.argmax(preds, dim=-1)
        for cls in range(self.num_classes):
            mask = cls == labels
            self.true_positive[cls] += ((preds == cls) & mask).sum().item()
            self.false_positive[cls] += ((preds == cls) & ~mask).sum().item()
            self.false_negative[cls] += ((preds != cls) & mask).sum().item()

    def compute(self) ->  float:
        """
        Compute the F1 score based on the mode specified during initialization.

        Returns:
            float: The computed F1 score.
        """
        if self.mode == "micro":
            tp_sum = self.true_positive.sum()
            fn_sum = self.false_negative.sum()
            fp_sum = self.false_positive.sum()
            recall =  (tp_sum / (tp_sum + fn_sum + 1e-10)).item()
            precision = (tp_sum / (tp_sum + fp_sum + 1e-10)).item()
            return 2 * recall * precision / (recall + precision + 1e-10)
        else:
            recall = (self.true_positive / (self.true_positive + self.false_negative + 1e-10))
            precision = (self.true_positive / (self.true_positive + self.false_positive + 1e-10))
            return (2 * recall * precision / (recall + precision + 1e-10)).mean().item()

    def reset(self) -> None:
        """
        Reset the state of the metrics (true positives, false positives, false negatives) to initial state.
        """
        self.true_positive = None
        self.false_positive = None
        self.false_negative = None
        self.initialized = False