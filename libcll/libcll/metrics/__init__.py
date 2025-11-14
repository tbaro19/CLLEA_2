"""Metrics module for behavior descriptors and quality diversity analysis."""

from .error_asymmetry import (
    compute_error_asymmetry,
    compute_per_class_accuracy,
    compute_accuracy_variance,
    compute_class_accuracy_disparity,
    ErrorAsymmetryMetrics,
    compute_error_asymmetry_from_logits,
    format_error_asymmetry_report,
)
from .visualize import (
    plot_qd_map_2d,
    plot_qd_heatmap,
    plot_per_class_accuracy_bars,
    plot_error_asymmetry_evolution,
    create_qd_report,
)

__all__ = [
    # Core metrics
    "compute_error_asymmetry",
    "compute_error_asymmetry_from_logits",
    "compute_per_class_accuracy",
    "compute_accuracy_variance",
    "compute_class_accuracy_disparity",
    "ErrorAsymmetryMetrics",
    "format_error_asymmetry_report",
    # Visualization
    "plot_qd_map_2d",
    "plot_qd_heatmap",
    "plot_per_class_accuracy_bars",
    "plot_error_asymmetry_evolution",
    "create_qd_report",
]
