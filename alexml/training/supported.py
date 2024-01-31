import alexml.training.metrics as metrics

supported_metrics = {
    "accuracy": metrics.Accuracy(),
    "recall": metrics.Recall(), # micro by default
    "micro-recall": metrics.Recall("micro"),
    "macro-recall": metrics.Recall("macro"),
    "precision": metrics.Precision(), # micro by default
    "micro-precision": metrics.Precision("micro"),
    "macro-precision": metrics.Precision("macro"),
    "f1": metrics.F1Score(), # micro by default
    "macro-f1": metrics.F1Score("macro"),
    "micro-f1": metrics.F1Score("micro"),
}