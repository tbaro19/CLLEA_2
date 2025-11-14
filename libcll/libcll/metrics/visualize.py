"""Visualization utilities for Quality Diversity maps and Error Asymmetry."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import seaborn as sns


def plot_qd_map_2d(
    accuracy: np.ndarray,
    behavior_descriptor: np.ndarray,
    values: Optional[np.ndarray] = None,
    xlabel: str = "Overall Accuracy",
    ylabel: str = "Accuracy Variance (BD)",
    title: str = "Quality Diversity Map: Error Asymmetry",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    cmap: str = "viridis",
    alpha: float = 0.6,
) -> plt.Figure:
    """Plot a 2D Quality Diversity map.
    
    This creates a scatter plot showing the relationship between an objective
    (e.g., overall accuracy) and a behavior descriptor (e.g., accuracy variance).
    
    Parameters
    ----------
    accuracy : np.ndarray
        Overall accuracy values (x-axis)
    behavior_descriptor : np.ndarray
        Behavior descriptor values (y-axis)
    values : np.ndarray, optional
        Values to use for coloring points. If None, uses accuracy.
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
    cmap : str
        Colormap name
    alpha : float
        Transparency of points
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if values is None:
        values = accuracy
    
    scatter = ax.scatter(
        accuracy,
        behavior_descriptor,
        c=values,
        cmap=cmap,
        alpha=alpha,
        s=100,
        edgecolors='black',
        linewidth=0.5
    )
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Objective Value', fontsize=10)
    
    # Add interpretation regions
    # Low variance = Generalist, High variance = Specialist
    if "variance" in ylabel.lower() or "disparity" in ylabel.lower():
        y_mid = (behavior_descriptor.max() + behavior_descriptor.min()) / 2
        ax.axhline(y=y_mid, color='red', linestyle='--', alpha=0.3, linewidth=1)
        ax.text(
            accuracy.min(), y_mid * 1.1,
            'Threshold: Generalist ← → Specialist',
            fontsize=9, color='red', alpha=0.7
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"QD map saved to: {save_path}")
    
    return fig


def plot_qd_heatmap(
    accuracy: np.ndarray,
    behavior_descriptor: np.ndarray,
    values: np.ndarray,
    bins: Tuple[int, int] = (20, 20),
    xlabel: str = "Overall Accuracy",
    ylabel: str = "Accuracy Variance (BD)",
    title: str = "QD Heatmap: Error Asymmetry",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    cmap: str = "YlOrRd",
    aggregation: str = "max",
) -> plt.Figure:
    """Plot a heatmap for Quality Diversity archive.
    
    This creates a 2D heatmap showing the best performance achieved in different
    regions of the behavior space.
    
    Parameters
    ----------
    accuracy : np.ndarray
        Overall accuracy values
    behavior_descriptor : np.ndarray
        Behavior descriptor values
    values : np.ndarray
        Values to aggregate in each bin
    bins : tuple
        Number of bins for (x, y) dimensions
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
    cmap : str
        Colormap name
    aggregation : str
        Aggregation method ('max', 'mean', 'min')
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create 2D bins
    x_bins = np.linspace(accuracy.min(), accuracy.max(), bins[0] + 1)
    y_bins = np.linspace(behavior_descriptor.min(), behavior_descriptor.max(), bins[1] + 1)
    
    # Initialize grid
    grid = np.full((bins[1], bins[0]), np.nan)
    
    # Aggregate values in each bin
    for i in range(len(accuracy)):
        x_idx = np.digitize(accuracy[i], x_bins) - 1
        y_idx = np.digitize(behavior_descriptor[i], y_bins) - 1
        
        if 0 <= x_idx < bins[0] and 0 <= y_idx < bins[1]:
            if np.isnan(grid[y_idx, x_idx]):
                grid[y_idx, x_idx] = values[i]
            else:
                if aggregation == "max":
                    grid[y_idx, x_idx] = max(grid[y_idx, x_idx], values[i])
                elif aggregation == "mean":
                    grid[y_idx, x_idx] = (grid[y_idx, x_idx] + values[i]) / 2
                elif aggregation == "min":
                    grid[y_idx, x_idx] = min(grid[y_idx, x_idx], values[i])
    
    # Plot heatmap
    im = ax.imshow(
        grid,
        origin='lower',
        aspect='auto',
        cmap=cmap,
        extent=[accuracy.min(), accuracy.max(), 
                behavior_descriptor.min(), behavior_descriptor.max()],
        interpolation='nearest'
    )
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Best Objective Value', fontsize=10)
    
    # Add coverage information
    coverage = np.sum(~np.isnan(grid)) / (bins[0] * bins[1]) * 100
    ax.text(
        0.02, 0.98, f'Coverage: {coverage:.1f}%',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"QD heatmap saved to: {save_path}")
    
    return fig


def plot_per_class_accuracy_bars(
    per_class_accuracy: np.ndarray,
    class_counts: np.ndarray,
    title: str = "Per-Class Accuracy",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot bar chart of per-class accuracies.
    
    Parameters
    ----------
    per_class_accuracy : np.ndarray
        Accuracy for each class
    class_counts : np.ndarray
        Number of samples for each class
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    num_classes = len(per_class_accuracy)
    classes = np.arange(num_classes)
    
    # Filter out NaN values
    valid_mask = ~np.isnan(per_class_accuracy)
    
    colors = ['green' if acc > 0.7 else 'orange' if acc > 0.5 else 'red' 
              for acc in per_class_accuracy]
    
    bars = ax.bar(classes[valid_mask], per_class_accuracy[valid_mask], 
                  color=np.array(colors)[valid_mask], alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(classes)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    # Add sample counts as text
    for i, (acc, count) in enumerate(zip(per_class_accuracy, class_counts)):
        if not np.isnan(acc):
            ax.text(i, acc + 0.02, f'n={int(count)}', 
                   ha='center', va='bottom', fontsize=8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Good (>70%)'),
        Patch(facecolor='orange', alpha=0.7, label='Medium (50-70%)'),
        Patch(facecolor='red', alpha=0.7, label='Poor (<50%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class accuracy plot saved to: {save_path}")
    
    return fig


def plot_error_asymmetry_evolution(
    epochs: List[int],
    accuracy: List[float],
    variance: List[float],
    disparity: List[float],
    title: str = "Error Asymmetry Evolution",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the evolution of error asymmetry metrics over training.
    
    Parameters
    ----------
    epochs : list
        Epoch numbers
    accuracy : list
        Overall accuracy at each epoch
    variance : list
        Accuracy variance at each epoch
    disparity : list
        Max disparity at each epoch
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot accuracy
    axes[0].plot(epochs, accuracy, marker='o', linewidth=2, markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Overall Accuracy', fontsize=10)
    axes[0].set_title('Overall Accuracy', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot variance
    axes[1].plot(epochs, variance, marker='s', linewidth=2, markersize=4, color='orange')
    axes[1].set_xlabel('Epoch', fontsize=10)
    axes[1].set_ylabel('Accuracy Variance', fontsize=10)
    axes[1].set_title('Accuracy Variance (BD)', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot disparity
    axes[2].plot(epochs, disparity, marker='^', linewidth=2, markersize=4, color='red')
    axes[2].set_xlabel('Epoch', fontsize=10)
    axes[2].set_ylabel('Max Disparity', fontsize=10)
    axes[2].set_title('Max Class Disparity (BD)', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evolution plot saved to: {save_path}")
    
    return fig


def create_qd_report(
    metrics_dict: Dict[str, float],
    per_class_accuracy: np.ndarray,
    class_counts: np.ndarray,
    output_dir: str,
    prefix: str = "qd_report",
) -> None:
    """Create a comprehensive QD report with multiple visualizations.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary containing error asymmetry metrics
    per_class_accuracy : np.ndarray
        Per-class accuracy values
    class_counts : np.ndarray
        Number of samples per class
    output_dir : str
        Directory to save the report
    prefix : str
        Prefix for output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create per-class accuracy plot
    plot_per_class_accuracy_bars(
        per_class_accuracy,
        class_counts,
        title=f"Per-Class Accuracy - {prefix}",
        save_path=str(output_path / f"{prefix}_per_class_accuracy.png")
    )
    
    # Create summary report
    report_lines = [
        "=" * 70,
        f"Quality Diversity Report: {prefix}",
        "=" * 70,
        "",
        "Objective Metrics:",
        f"  Overall Accuracy: {metrics_dict.get('accuracy', 0):.4f}",
        "",
        "Behavior Descriptors:",
        f"  Accuracy Variance:  {metrics_dict.get('variance', 0):.6f}",
        f"  Accuracy Std Dev:   {metrics_dict.get('std', 0):.6f}",
        f"  Max Disparity:      {metrics_dict.get('max_disparity', 0):.4f}",
        "",
        "Class Performance:",
        f"  Min Class Accuracy: {metrics_dict.get('min_accuracy', 0):.4f}",
        f"  Max Class Accuracy: {metrics_dict.get('max_accuracy', 0):.4f}",
        "",
        "=" * 70,
    ]
    
    report_text = "\n".join(report_lines)
    
    # Save report
    with open(output_path / f"{prefix}_summary.txt", "w") as f:
        f.write(report_text)
    
    print(f"QD report saved to: {output_path}")
    print(report_text)
