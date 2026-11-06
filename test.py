"""
Module: test.py

This module provides functions for evaluating a video classification model on test data.
It includes functions to compute predictions and accuracy, generate a detailed classification report,
and compute a multilabel confusion matrix for all classes.

Functions:
    - test: Evaluates the model on a test DataLoader and returns the ground truth labels,
      predicted labels, and overall accuracy.
    - get_test_report: Generates a classification report using scikit-learn's classification_report.
    - get_confusion_matrix: Computes a multilabel confusion matrix for each class.
"""

import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, multilabel_confusion_matrix

@torch.no_grad()
def test(model, dataloader, device, wandb_run=None):
    """Return test labels, predictions, and accuracy, and optionally log metrics."""
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    targets, outputs = [], []

    for inputs, labels in tqdm(dataloader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        total_loss += loss_function(logits, labels).item()
        outputs.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.cpu().tolist())

    if not targets:
        raise ValueError("The test loader returned no clips.")

    correct = sum(prediction == target for prediction, target in zip(outputs, targets))
    accuracy = correct / len(targets)
    if wandb_run is not None:
        wandb_run.log({
            "test/loss": total_loss / len(targets),
            "test/accuracy_pct": 100 * accuracy,
            "test/correct": correct,
            "test/clips": len(targets),
        })
    return targets, outputs, accuracy

def get_test_report(target, output, target_names):
    """
    Generate a detailed classification report based on test results.
    
    This function uses scikit-learn's classification_report to produce a dictionary
    containing precision, recall, F1-score, and support for each class.
    
    Args:
        target (list): Ground truth labels.
        output (list): Predicted labels.
        target_names (list): List of class names corresponding to the labels.
    
    Returns:
        dict: A classification report as a dictionary.
    """
    return classification_report(target, output, output_dict=True, target_names=target_names)

def get_confusion_matrix(targets, outputs, labels_dict, all_cats):
    """
    Compute a multilabel confusion matrix for each class.
    
    This function converts numeric labels to their corresponding class names using the provided
    labels_dict, then computes a multilabel confusion matrix for each class using scikit-learn.
    
    Args:
        targets (list): Ground truth numeric labels.
        outputs (list): Predicted numeric labels.
        labels_dict (dict): Dictionary mapping class names to numeric labels.
        all_cats (list): List of all class names.
    
    Returns:
        dict: A dictionary where keys are class names and values are the corresponding confusion matrices.
    """
    # Create an inverse mapping from numeric label to class name
    inv_labels_dict = {label: cat for cat, label in labels_dict.items()}
    target_cats = [inv_labels_dict[target] for target in targets]
    output_cats = [inv_labels_dict[output] for output in outputs]
    confusion_mat = multilabel_confusion_matrix(target_cats, output_cats, labels=all_cats)
    return {label: mat for label, mat in zip(all_cats, confusion_mat)}
