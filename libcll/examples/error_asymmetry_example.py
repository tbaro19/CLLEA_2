"""Example script demonstrating Error Asymmetry BD metrics usage.

This script shows how to:
1. Compute error asymmetry metrics during or after training
2. Visualize QD maps with [Overall Accuracy] vs [Accuracy Variance]
3. Generate comprehensive reports on model behavior
"""

import torch
import numpy as np
from libcll.metrics.error_asymmetry import (
    compute_error_asymmetry,
    format_error_asymmetry_report,
)
from libcll.metrics.visualize import (
    plot_qd_map_2d,
    plot_qd_heatmap,
    plot_per_class_accuracy_bars,
    plot_error_asymmetry_evolution,
    create_qd_report,
)


def example_compute_metrics():
    """Example: Computing error asymmetry metrics from predictions."""
    print("=" * 70)
    print("Example 1: Computing Error Asymmetry Metrics")
    print("=" * 70)
    
    # Simulate predictions and targets
    # Scenario: Biased model (good at classes 0-1, poor at class 2)
    predictions = torch.tensor([0, 0, 1, 1, 1, 2, 0, 1, 0, 1, 0, 1, 2, 2, 0])
    targets = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 0, 1, 2, 2, 0])
    num_classes = 3
    
    # Compute metrics
    metrics = compute_error_asymmetry(predictions, targets, num_classes)
    
    # Print formatted report
    print(format_error_asymmetry_report(metrics))
    print()


def example_visualize_qd_map():
    """Example: Visualizing QD map with multiple model configurations."""
    print("=" * 70)
    print("Example 2: Visualizing Quality Diversity Map")
    print("=" * 70)
    
    # Simulate multiple model runs with different characteristics
    np.random.seed(42)
    n_models = 50
    
    # Generate diverse model behaviors
    # Generalists: High accuracy, low variance
    n_generalists = 20
    acc_generalist = np.random.uniform(0.75, 0.95, n_generalists)
    var_generalist = np.random.uniform(0.001, 0.02, n_generalists)
    
    # Specialists: Medium-high accuracy, high variance
    n_specialists = 20
    acc_specialist = np.random.uniform(0.65, 0.85, n_specialists)
    var_specialist = np.random.uniform(0.05, 0.15, n_specialists)
    
    # Poor models: Low accuracy, variable variance
    n_poor = 10
    acc_poor = np.random.uniform(0.3, 0.6, n_poor)
    var_poor = np.random.uniform(0.02, 0.1, n_poor)
    
    # Combine all models
    accuracy = np.concatenate([acc_generalist, acc_specialist, acc_poor])
    variance = np.concatenate([var_generalist, var_specialist, var_poor])
    
    # Plot QD map
    print("Creating QD scatter plot...")
    fig = plot_qd_map_2d(
        accuracy=accuracy,
        behavior_descriptor=variance,
        xlabel="Overall Accuracy",
        ylabel="Accuracy Variance (Error Asymmetry BD)",
        title="QD Map: Generalist vs Specialist Models",
        save_path="example_qd_map.png"
    )
    
    # Plot QD heatmap
    print("Creating QD heatmap...")
    fig = plot_qd_heatmap(
        accuracy=accuracy,
        behavior_descriptor=variance,
        values=accuracy,  # Use accuracy as the objective
        bins=(15, 15),
        xlabel="Overall Accuracy",
        ylabel="Accuracy Variance (Error Asymmetry BD)",
        title="QD Heatmap: Archive Coverage",
        save_path="example_qd_heatmap.png"
    )
    
    print("Visualizations saved!")
    print()


def example_per_class_analysis():
    """Example: Analyzing per-class performance."""
    print("=" * 70)
    print("Example 3: Per-Class Accuracy Analysis")
    print("=" * 70)
    
    # Simulate per-class accuracies for different scenarios
    
    # Scenario 1: Balanced model
    print("\nScenario 1: Balanced Model (Generalist)")
    per_class_acc_balanced = np.array([0.92, 0.89, 0.91, 0.90, 0.88])
    class_counts = np.array([100, 100, 100, 100, 100])
    
    fig = plot_per_class_accuracy_bars(
        per_class_acc_balanced,
        class_counts,
        title="Balanced Model - Per-Class Accuracy",
        save_path="example_balanced_per_class.png"
    )
    print(f"Mean Accuracy: {per_class_acc_balanced.mean():.4f}")
    print(f"Variance: {per_class_acc_balanced.var():.6f}")
    
    # Scenario 2: Biased model
    print("\nScenario 2: Biased Model (Specialist)")
    per_class_acc_biased = np.array([0.95, 0.93, 0.45, 0.48, 0.40])
    
    fig = plot_per_class_accuracy_bars(
        per_class_acc_biased,
        class_counts,
        title="Biased Model - Per-Class Accuracy",
        save_path="example_biased_per_class.png"
    )
    print(f"Mean Accuracy: {per_class_acc_biased.mean():.4f}")
    print(f"Variance: {per_class_acc_biased.var():.6f}")
    print()


def example_training_evolution():
    """Example: Tracking error asymmetry evolution during training."""
    print("=" * 70)
    print("Example 4: Error Asymmetry Evolution During Training")
    print("=" * 70)
    
    # Simulate training progress
    epochs = list(range(0, 101, 10))
    
    # Scenario: Model starts balanced but becomes biased
    accuracy = [0.40, 0.55, 0.68, 0.75, 0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.92]
    variance = [0.01, 0.015, 0.025, 0.040, 0.055, 0.070, 0.080, 0.090, 0.095, 0.098, 0.100]
    disparity = [0.10, 0.15, 0.20, 0.28, 0.35, 0.42, 0.48, 0.52, 0.55, 0.57, 0.58]
    
    fig = plot_error_asymmetry_evolution(
        epochs=epochs,
        accuracy=accuracy,
        variance=variance,
        disparity=disparity,
        title="Model Training: Generalist → Specialist Transition",
        save_path="example_evolution.png"
    )
    
    print("Evolution plot created!")
    print("\nObservation:")
    print("  - Accuracy improves over training (good)")
    print("  - Variance and disparity increase (model becomes specialized)")
    print("  - Trade-off between overall performance and fairness")
    print()


def example_qd_report():
    """Example: Creating comprehensive QD report."""
    print("=" * 70)
    print("Example 5: Comprehensive QD Report")
    print("=" * 70)
    
    # Simulate model metrics
    metrics_dict = {
        'accuracy': 0.87,
        'variance': 0.045,
        'std': 0.212,
        'max_disparity': 0.35,
        'min_accuracy': 0.68,
        'max_accuracy': 0.96,
    }
    
    per_class_accuracy = np.array([0.96, 0.92, 0.85, 0.78, 0.68])
    class_counts = np.array([120, 115, 110, 105, 100])
    
    create_qd_report(
        metrics_dict=metrics_dict,
        per_class_accuracy=per_class_accuracy,
        class_counts=class_counts,
        output_dir="qd_reports",
        prefix="example_model"
    )
    
    print()


def example_integration_with_training():
    """Example: How to integrate with PyTorch Lightning training."""
    print("=" * 70)
    print("Example 6: Integration with Training Loop")
    print("=" * 70)
    
    code_example = '''
    # In your PyTorch Lightning Strategy class:
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        
        # Your existing validation logic
        val_loss = self.compute_loss(out, y)
        
        # Store predictions for error asymmetry
        y_pred = torch.argmax(out, dim=1)
        self.val_predictions.append(y_pred)
        self.val_targets.append(y)
        
        return {"val_loss": val_loss}
    
    def on_validation_epoch_end(self):
        # Compute error asymmetry metrics
        all_predictions = torch.cat(self.val_predictions)
        all_targets = torch.cat(self.val_targets)
        
        from libcll.metrics import compute_error_asymmetry
        ea_metrics = compute_error_asymmetry(
            all_predictions, all_targets, self.num_classes
        )
        
        # Log to tensorboard
        self.log("Valid_EA_Accuracy", ea_metrics.overall_accuracy)
        self.log("Valid_EA_Variance", ea_metrics.accuracy_variance)
        self.log("Valid_EA_MaxDisparity", ea_metrics.max_disparity)
        
        # Clear buffers
        self.val_predictions.clear()
        self.val_targets.clear()
    '''
    
    print(code_example)
    print()


def main():
    """Run all examples."""
    print("\n")
    print("*" * 70)
    print("Error Asymmetry BD Metrics - Examples")
    print("*" * 70)
    print()
    
    # Run examples
    example_compute_metrics()
    example_visualize_qd_map()
    example_per_class_analysis()
    example_training_evolution()
    example_qd_report()
    example_integration_with_training()
    
    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - example_qd_map.png")
    print("  - example_qd_heatmap.png")
    print("  - example_balanced_per_class.png")
    print("  - example_biased_per_class.png")
    print("  - example_evolution.png")
    print("  - qd_reports/example_model_*")
    print()


if __name__ == "__main__":
    main()
