"""Error Asymmetry: Behavior Descriptor for Class Imbalance in Continual Learning.

In continual learning (CLL), one major challenge is that the model may learn to become
biased toward certain classes. This module provides metrics to measure the performance
disparity between classes, which can be used as behavior descriptors (BD) in Quality
Diversity (QD) optimization.

Behavior Descriptors:
---------------------
1. Accuracy Variance: Variance of accuracy across all classes
   - BD close to 0: Fair model (equal performance across classes)
   - BD far from 0: Biased model (specialist behavior)

2. Class Accuracy Disparity: Maximum difference between class accuracies
   - BD close to 0: Balanced performance
   - BD far from 0: Strong specialization

QD Map Interpretation:
---------------------
- X-axis: Overall Accuracy (objective to maximize)
- Y-axis: Accuracy Variance or Disparity (behavior descriptor)
- Low variance + high accuracy: Generalist models
- High variance + high accuracy: Specialist models (good on some classes)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple


class ErrorAsymmetryMetrics(NamedTuple):
    """Container for error asymmetry metrics.
    
    Attributes:
        overall_accuracy: Overall accuracy across all classes (FITNESS - to maximize)
        per_class_accuracy: Accuracy for each individual class
        accuracy_variance: Variance of per-class accuracies (BD - no clear good/bad)
        accuracy_std: Standard deviation of per-class accuracies (BD - no clear good/bad)
        max_disparity: Maximum difference between any two class accuracies (BD)
        min_accuracy: Minimum accuracy across all classes
        max_accuracy: Maximum accuracy across all classes
        performance_entropy: Entropy of per-class accuracies (BD - higher = more uniform)
        prediction_concentration: Concentration of predictions (BD - class preference)
        class_imbalance_adaptation: How well model adapts to imbalanced data (BD)
        specialization_index: Degree of specialization vs generalization (BD)
        class_counts: Number of samples per class
        num_classes: Total number of classes
    """
    overall_accuracy: float
    per_class_accuracy: np.ndarray
    accuracy_variance: float
    accuracy_std: float
    max_disparity: float
    min_accuracy: float
    max_accuracy: float
    performance_entropy: float
    prediction_concentration: float
    class_imbalance_adaptation: float
    specialization_index: float
    class_counts: np.ndarray
    num_classes: int


def compute_per_class_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-class accuracy.
    
    Parameters
    ----------
    predictions : torch.Tensor
        Predicted class labels, shape (N,)
    targets : torch.Tensor
        True class labels, shape (N,)
    num_classes : int
        Total number of classes
        
    Returns
    -------
    per_class_acc : np.ndarray
        Accuracy for each class, shape (num_classes,)
    class_counts : np.ndarray
        Number of samples for each class, shape (num_classes,)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    per_class_acc = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)
    
    for class_idx in range(num_classes):
        # Find samples belonging to this class
        class_mask = targets == class_idx
        class_counts[class_idx] = class_mask.sum()
        
        if class_counts[class_idx] > 0:
            # Calculate accuracy for this class
            class_correct = (predictions[class_mask] == targets[class_mask]).sum()
            per_class_acc[class_idx] = class_correct / class_counts[class_idx]
        else:
            # No samples for this class, set accuracy to NaN
            per_class_acc[class_idx] = np.nan
    
    return per_class_acc, class_counts


def compute_accuracy_variance(per_class_accuracy: np.ndarray) -> Tuple[float, float]:
    """Compute variance and standard deviation of per-class accuracies.
    
    Parameters
    ----------
    per_class_accuracy : np.ndarray
        Accuracy for each class, shape (num_classes,)
        
    Returns
    -------
    variance : float
        Variance of per-class accuracies
    std : float
        Standard deviation of per-class accuracies
    """
    # Filter out NaN values (classes with no samples)
    valid_accuracies = per_class_accuracy[~np.isnan(per_class_accuracy)]
    
    if len(valid_accuracies) == 0:
        return 0.0, 0.0
    
    variance = np.var(valid_accuracies)
    std = np.std(valid_accuracies)
    
    return float(variance), float(std)


def compute_class_accuracy_disparity(per_class_accuracy: np.ndarray) -> float:
    """Compute maximum disparity between class accuracies.
    
    This is defined as: max(accuracy) - min(accuracy) across all classes.
    
    Parameters
    ----------
    per_class_accuracy : np.ndarray
        Accuracy for each class, shape (num_classes,)
        
    Returns
    -------
    disparity : float
        Maximum difference between class accuracies
    """
    # Filter out NaN values (classes with no samples)
    valid_accuracies = per_class_accuracy[~np.isnan(per_class_accuracy)]
    
    if len(valid_accuracies) == 0:
        return 0.0
    
    max_acc = np.max(valid_accuracies)
    min_acc = np.min(valid_accuracies)
    disparity = max_acc - min_acc
    
    return float(disparity)


def compute_performance_entropy(per_class_accuracy: np.ndarray) -> float:
    """Compute entropy of per-class accuracy distribution.
    
    Higher entropy indicates more uniform performance across classes.
    This is a BD - we don't know if higher or lower is better.
    
    Parameters
    ----------
    per_class_accuracy : np.ndarray
        Accuracy for each class, shape (num_classes,)
        
    Returns
    -------
    entropy : float
        Entropy of the accuracy distribution
    """
    valid_accuracies = per_class_accuracy[~np.isnan(per_class_accuracy)]
    
    if len(valid_accuracies) == 0:
        return 0.0
    
    # Normalize to probability distribution
    # Add small epsilon to avoid log(0)
    acc_sum = valid_accuracies.sum() + 1e-10
    probs = (valid_accuracies + 1e-10) / acc_sum
    
    # Compute entropy: -sum(p * log(p))
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    return float(entropy)


def compute_prediction_concentration(predictions: np.ndarray, num_classes: int) -> float:
    """Compute concentration of predictions across classes.
    
    Measures how concentrated predictions are on certain classes.
    This is a BD - balanced concentration might be good or bad depending on true distribution.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted class labels
    num_classes : int
        Total number of classes
        
    Returns
    -------
    concentration : float
        Concentration metric (Gini coefficient style)
    """
    # Count predictions per class
    pred_counts = np.zeros(num_classes)
    for pred in predictions:
        pred_counts[int(pred)] += 1
    
    # Normalize
    total = pred_counts.sum()
    if total == 0:
        return 0.0
    
    pred_probs = pred_counts / total
    
    # Compute concentration using variance
    # Higher value = more concentrated on fewer classes
    concentration = float(np.var(pred_probs) * num_classes)
    
    return concentration


def compute_class_imbalance_adaptation(per_class_accuracy: np.ndarray, class_counts: np.ndarray) -> float:
    """Compute how well the model adapts to class imbalance.
    
    Measures correlation between class frequency and accuracy.
    Positive = better on frequent classes, Negative = better on rare classes.
    This is a BD - either direction could be desirable depending on goals.
    
    Parameters
    ----------
    per_class_accuracy : np.ndarray
        Accuracy for each class
    class_counts : np.ndarray
        Number of samples per class
        
    Returns
    -------
    adaptation : float
        Correlation between class frequency and accuracy (-1 to 1)
    """
    valid_mask = ~np.isnan(per_class_accuracy)
    valid_acc = per_class_accuracy[valid_mask]
    valid_counts = class_counts[valid_mask]
    
    if len(valid_acc) < 2:
        return 0.0
    
    # Compute Pearson correlation
    if np.std(valid_acc) == 0 or np.std(valid_counts) == 0:
        return 0.0
    
    correlation = np.corrcoef(valid_counts, valid_acc)[0, 1]
    
    return float(correlation) if not np.isnan(correlation) else 0.0


def compute_specialization_index(per_class_accuracy: np.ndarray) -> float:
    """Compute specialization index of the model.
    
    Measures tendency toward specialist vs generalist behavior.
    High values indicate specialist (good at few classes).
    Low values indicate generalist (decent at all classes).
    This is a BD - neither is inherently better.
    
    Parameters
    ----------
    per_class_accuracy : np.ndarray
        Accuracy for each class
        
    Returns
    -------
    specialization : float
        Specialization index (0 to 1)
    """
    valid_accuracies = per_class_accuracy[~np.isnan(per_class_accuracy)]
    
    if len(valid_accuracies) == 0:
        return 0.0
    
    # Coefficient of variation: std / mean
    # High CV = specialist, Low CV = generalist
    mean_acc = valid_accuracies.mean()
    if mean_acc == 0:
        return 0.0
    
    specialization = float(valid_accuracies.std() / mean_acc)
    
    return specialization


def compute_error_asymmetry(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> ErrorAsymmetryMetrics:
    """Compute comprehensive error asymmetry metrics.
    
    This function computes all error asymmetry behavior descriptors including:
    - Overall accuracy
    - Per-class accuracy
    - Accuracy variance and standard deviation
    - Maximum class accuracy disparity
    - Min/max class accuracies
    
    Parameters
    ----------
    predictions : torch.Tensor
        Predicted class labels, shape (N,)
    targets : torch.Tensor
        True class labels, shape (N,)
    num_classes : int
        Total number of classes
        
    Returns
    -------
    metrics : ErrorAsymmetryMetrics
        Named tuple containing all error asymmetry metrics
        
    Examples
    --------
    >>> predictions = torch.tensor([0, 1, 1, 2, 2, 2])
    >>> targets = torch.tensor([0, 1, 2, 2, 2, 2])
    >>> metrics = compute_error_asymmetry(predictions, targets, num_classes=3)
    >>> print(f"Overall Accuracy: {metrics.overall_accuracy:.2f}")
    >>> print(f"Accuracy Variance: {metrics.accuracy_variance:.4f}")
    >>> print(f"Max Disparity: {metrics.max_disparity:.2f}")
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Compute overall accuracy
    overall_accuracy = float((predictions == targets).mean())
    
    # Compute per-class accuracy
    per_class_acc, class_counts = compute_per_class_accuracy(
        torch.from_numpy(predictions),
        torch.from_numpy(targets),
        num_classes
    )
    
    # Compute variance and std
    variance, std = compute_accuracy_variance(per_class_acc)
    
    # Compute disparity
    disparity = compute_class_accuracy_disparity(per_class_acc)
    
    # Get min and max accuracies
    valid_accuracies = per_class_acc[~np.isnan(per_class_acc)]
    min_accuracy = float(np.min(valid_accuracies)) if len(valid_accuracies) > 0 else 0.0
    max_accuracy = float(np.max(valid_accuracies)) if len(valid_accuracies) > 0 else 0.0
    
    # Compute additional behavior descriptors
    performance_entropy = compute_performance_entropy(per_class_acc)
    prediction_concentration = compute_prediction_concentration(predictions, num_classes)
    class_imbalance_adaptation = compute_class_imbalance_adaptation(per_class_acc, class_counts)
    specialization_index = compute_specialization_index(per_class_acc)
    
    return ErrorAsymmetryMetrics(
        overall_accuracy=overall_accuracy,
        per_class_accuracy=per_class_acc,
        accuracy_variance=variance,
        accuracy_std=std,
        max_disparity=disparity,
        min_accuracy=min_accuracy,
        max_accuracy=max_accuracy,
        performance_entropy=performance_entropy,
        prediction_concentration=prediction_concentration,
        class_imbalance_adaptation=class_imbalance_adaptation,
        specialization_index=specialization_index,
        class_counts=class_counts,
        num_classes=num_classes,
    )


def compute_error_asymmetry_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> ErrorAsymmetryMetrics:
    """Compute error asymmetry metrics from model logits.
    
    This is a convenience function that converts logits to predictions
    and then computes error asymmetry metrics.
    
    Parameters
    ----------
    logits : torch.Tensor
        Model output logits, shape (N, num_classes)
    targets : torch.Tensor
        True class labels, shape (N,)
    num_classes : int
        Total number of classes
        
    Returns
    -------
    metrics : ErrorAsymmetryMetrics
        Named tuple containing all error asymmetry metrics
    """
    predictions = torch.argmax(logits, dim=1)
    return compute_error_asymmetry(predictions, targets, num_classes)


def format_error_asymmetry_report(metrics: ErrorAsymmetryMetrics) -> str:
    """Format error asymmetry metrics into a human-readable report.
    
    Parameters
    ----------
    metrics : ErrorAsymmetryMetrics
        Error asymmetry metrics to format
        
    Returns
    -------
    report : str
        Formatted report string
    """
    report_lines = [
        "=" * 70,
        "Error Asymmetry Report",
        "=" * 70,
        f"Overall Accuracy: {metrics.overall_accuracy:.4f} ({metrics.overall_accuracy * 100:.2f}%)",
        f"Number of Classes: {metrics.num_classes}",
        "",
        "Behavior Descriptors:",
        f"  Accuracy Variance: {metrics.accuracy_variance:.6f}",
        f"  Accuracy Std Dev:  {metrics.accuracy_std:.6f}",
        f"  Max Disparity:     {metrics.max_disparity:.4f} ({metrics.max_disparity * 100:.2f}%)",
        "",
        "Class Performance Range:",
        f"  Min Accuracy: {metrics.min_accuracy:.4f} ({metrics.min_accuracy * 100:.2f}%)",
        f"  Max Accuracy: {metrics.max_accuracy:.4f} ({metrics.max_accuracy * 100:.2f}%)",
        "",
        "Per-Class Accuracy:",
    ]
    
    for class_idx in range(metrics.num_classes):
        acc = metrics.per_class_accuracy[class_idx]
        count = metrics.class_counts[class_idx]
        if np.isnan(acc):
            report_lines.append(f"  Class {class_idx}: N/A (no samples)")
        else:
            report_lines.append(
                f"  Class {class_idx}: {acc:.4f} ({acc * 100:.2f}%) "
                f"[{int(count)} samples]"
            )
    
    report_lines.append("=" * 70)
    
    # Interpretation
    if metrics.accuracy_variance < 0.01:
        interpretation = "GENERALIST (fair/balanced performance)"
    elif metrics.accuracy_variance > 0.05:
        interpretation = "SPECIALIST (biased toward certain classes)"
    else:
        interpretation = "MODERATE (some class preference)"
    
    report_lines.append(f"Interpretation: {interpretation}")
    report_lines.append("=" * 70)
    
    return "\n".join(report_lines)
