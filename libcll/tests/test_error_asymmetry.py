"""Unit tests for Error Asymmetry metrics."""

import torch
import numpy as np
import pytest
from libcll.metrics.error_asymmetry import (
    compute_error_asymmetry,
    compute_per_class_accuracy,
    compute_accuracy_variance,
    compute_class_accuracy_disparity,
    compute_error_asymmetry_from_logits,
    format_error_asymmetry_report,
)


class TestErrorAsymmetryMetrics:
    """Test suite for error asymmetry metrics."""
    
    def test_perfect_predictions(self):
        """Test with perfect predictions (all correct)."""
        predictions = torch.tensor([0, 1, 2, 0, 1, 2])
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        num_classes = 3
        
        metrics = compute_error_asymmetry(predictions, targets, num_classes)
        
        assert metrics.overall_accuracy == 1.0
        assert np.allclose(metrics.per_class_accuracy, [1.0, 1.0, 1.0])
        assert metrics.accuracy_variance == 0.0
        assert metrics.max_disparity == 0.0
        
    def test_balanced_accuracy(self):
        """Test with balanced per-class accuracies."""
        # 50% accuracy for all classes
        predictions = torch.tensor([0, 1, 1, 2, 2, 0])
        targets = torch.tensor([0, 0, 1, 1, 2, 2])
        num_classes = 3
        
        metrics = compute_error_asymmetry(predictions, targets, num_classes)
        
        # Each class: 1 correct out of 2
        assert metrics.overall_accuracy == 0.5
        assert np.allclose(metrics.per_class_accuracy, [0.5, 0.5, 0.5])
        assert metrics.accuracy_variance == 0.0
        assert metrics.max_disparity == 0.0
        
    def test_biased_model(self):
        """Test with biased model (good at some classes, poor at others)."""
        # Class 0: 100% correct, Class 1: 0% correct, Class 2: 100% correct
        predictions = torch.tensor([0, 0, 1, 1, 2, 2])
        targets = torch.tensor([0, 0, 0, 0, 2, 2])
        num_classes = 3
        
        metrics = compute_error_asymmetry(predictions, targets, num_classes)
        
        assert metrics.overall_accuracy == 4/6  # 4 correct out of 6
        assert metrics.per_class_accuracy[0] == 0.5  # 2 out of 4 correct
        assert metrics.per_class_accuracy[2] == 1.0  # 2 out of 2 correct
        assert metrics.accuracy_variance > 0  # Should have variance
        assert metrics.max_disparity > 0  # Should have disparity
        
    def test_per_class_accuracy(self):
        """Test per-class accuracy computation."""
        predictions = torch.tensor([0, 0, 1, 1, 2, 2])
        targets = torch.tensor([0, 1, 1, 2, 2, 0])
        num_classes = 3
        
        per_class_acc, class_counts = compute_per_class_accuracy(
            predictions, targets, num_classes
        )
        
        # Class 0: 1 correct out of 2
        assert per_class_acc[0] == 0.5
        assert class_counts[0] == 2
        
        # Class 1: 1 correct out of 2
        assert per_class_acc[1] == 0.5
        assert class_counts[1] == 2
        
        # Class 2: 1 correct out of 2
        assert per_class_acc[2] == 0.5
        assert class_counts[2] == 2
        
    def test_accuracy_variance(self):
        """Test accuracy variance computation."""
        # High variance case
        per_class_acc_high = np.array([0.9, 0.1, 0.9, 0.1])
        var_high, std_high = compute_accuracy_variance(per_class_acc_high)
        
        # Low variance case
        per_class_acc_low = np.array([0.5, 0.51, 0.49, 0.50])
        var_low, std_low = compute_accuracy_variance(per_class_acc_low)
        
        assert var_high > var_low
        assert std_high > std_low
        
    def test_class_accuracy_disparity(self):
        """Test max class disparity computation."""
        # Large disparity
        per_class_acc_large = np.array([0.9, 0.2, 0.8])
        disparity_large = compute_class_accuracy_disparity(per_class_acc_large)
        
        # Small disparity
        per_class_acc_small = np.array([0.5, 0.51, 0.52])
        disparity_small = compute_class_accuracy_disparity(per_class_acc_small)
        
        assert disparity_large > disparity_small
        assert disparity_large == 0.7  # 0.9 - 0.2
        assert disparity_small == pytest.approx(0.02, abs=0.01)
        
    def test_missing_classes(self):
        """Test handling of classes with no samples."""
        predictions = torch.tensor([0, 0, 1, 1])
        targets = torch.tensor([0, 0, 1, 1])
        num_classes = 5  # Classes 2, 3, 4 have no samples
        
        metrics = compute_error_asymmetry(predictions, targets, num_classes)
        
        assert metrics.overall_accuracy == 1.0
        assert metrics.per_class_accuracy[0] == 1.0
        assert metrics.per_class_accuracy[1] == 1.0
        assert np.isnan(metrics.per_class_accuracy[2])
        assert np.isnan(metrics.per_class_accuracy[3])
        assert np.isnan(metrics.per_class_accuracy[4])
        
    def test_from_logits(self):
        """Test computing metrics from logits."""
        # Create logits that will predict classes 0, 1, 2
        logits = torch.tensor([
            [3.0, 1.0, 0.0],  # Predicts 0
            [0.0, 3.0, 1.0],  # Predicts 1
            [1.0, 0.0, 3.0],  # Predicts 2
        ])
        targets = torch.tensor([0, 1, 2])
        num_classes = 3
        
        metrics = compute_error_asymmetry_from_logits(logits, targets, num_classes)
        
        assert metrics.overall_accuracy == 1.0
        
    def test_numpy_inputs(self):
        """Test that numpy arrays are handled correctly."""
        predictions = np.array([0, 1, 1, 2])
        targets = np.array([0, 1, 2, 2])
        num_classes = 3
        
        # Should work with numpy arrays
        metrics = compute_error_asymmetry(
            torch.from_numpy(predictions),
            torch.from_numpy(targets),
            num_classes
        )
        
        assert metrics.overall_accuracy == 0.75  # 3 out of 4 correct
        
    def test_format_report(self):
        """Test report formatting."""
        predictions = torch.tensor([0, 1, 2, 0, 1, 2])
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        num_classes = 3
        
        metrics = compute_error_asymmetry(predictions, targets, num_classes)
        report = format_error_asymmetry_report(metrics)
        
        assert "Overall Accuracy" in report
        assert "Accuracy Variance" in report
        assert "Per-Class Accuracy" in report
        assert "GENERALIST" in report  # Perfect model should be generalist
        
    def test_specialist_interpretation(self):
        """Test specialist model interpretation."""
        # Very biased model
        predictions = torch.tensor([0] * 10 + [1] * 10 + [0] * 10)
        targets = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
        num_classes = 3
        
        metrics = compute_error_asymmetry(predictions, targets, num_classes)
        report = format_error_asymmetry_report(metrics)
        
        # Should be classified as specialist due to high variance
        assert metrics.accuracy_variance > 0.05
        assert "SPECIALIST" in report


if __name__ == "__main__":
    # Run basic tests
    test = TestErrorAsymmetryMetrics()
    
    print("Running tests...")
    test.test_perfect_predictions()
    print("✓ test_perfect_predictions")
    
    test.test_balanced_accuracy()
    print("✓ test_balanced_accuracy")
    
    test.test_biased_model()
    print("✓ test_biased_model")
    
    test.test_per_class_accuracy()
    print("✓ test_per_class_accuracy")
    
    test.test_accuracy_variance()
    print("✓ test_accuracy_variance")
    
    test.test_class_accuracy_disparity()
    print("✓ test_class_accuracy_disparity")
    
    test.test_missing_classes()
    print("✓ test_missing_classes")
    
    test.test_from_logits()
    print("✓ test_from_logits")
    
    test.test_numpy_inputs()
    print("✓ test_numpy_inputs")
    
    test.test_format_report()
    print("✓ test_format_report")
    
    test.test_specialist_interpretation()
    print("✓ test_specialist_interpretation")
    
    print("\nAll tests passed! ✓")
