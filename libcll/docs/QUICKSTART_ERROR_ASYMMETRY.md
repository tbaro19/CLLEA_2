# Error Asymmetry BD Metrics - Quick Start Guide

## Installation

The Error Asymmetry metrics are now integrated into the libcll library. Make sure you have the required dependencies:

```bash
# Install libcll dependencies
cd libcll
pip install -r requirements.txt

# For visualization examples
pip install matplotlib seaborn

# For QD integration with pyribs
cd ../pyribs-master
pip install -e .
```

## 5-Minute Quick Start

### 1. Basic Usage - Compute Metrics

```python
import torch
from libcll.metrics import compute_error_asymmetry

# Your model predictions
predictions = torch.tensor([0, 1, 1, 2, 0, 1, 2, 2, 0, 1])
targets = torch.tensor([0, 1, 2, 2, 0, 1, 2, 1, 0, 1])
num_classes = 3

# Compute error asymmetry
metrics = compute_error_asymmetry(predictions, targets, num_classes)

print(f"Overall Accuracy: {metrics.overall_accuracy:.4f}")
print(f"Variance (BD): {metrics.accuracy_variance:.6f}")
print(f"Max Disparity: {metrics.max_disparity:.4f}")

# Interpretation
if metrics.accuracy_variance < 0.01:
    print("→ Generalist model (fair)")
elif metrics.accuracy_variance > 0.05:
    print("→ Specialist model (biased)")
else:
    print("→ Balanced model")
```

### 2. Automatic Integration with Training

Error Asymmetry metrics are **automatically tracked** when you use libcll strategies:

```python
from libcll.strategies import SCL
from libcll.models import build_model
from libcll.datasets import prepare_cl_data_module

# Your existing training code
model = build_model("dense", input_dim=784, hidden_dim=500, num_classes=10)
strategy = SCL(model=model, num_classes=10, lr=1e-4)

# Train as usual - Error Asymmetry is logged automatically!
trainer = pl.Trainer(max_epochs=100)
trainer.fit(strategy, train_loader, val_loader)

# Metrics are logged to TensorBoard:
# - Valid_EA_Accuracy
# - Valid_EA_Variance (BD)
# - Valid_EA_MaxDisparity (BD)
# - Test_EA_Class_0_Accuracy, Test_EA_Class_1_Accuracy, etc.
```

### 3. View Metrics in TensorBoard

```bash
tensorboard --logdir=outputs/your_experiment
```

Look for metrics starting with `Valid_EA_*` and `Test_EA_*`.

### 4. Create Visualizations

```python
from libcll.metrics.visualize import plot_per_class_accuracy_bars

# After testing
plot_per_class_accuracy_bars(
    per_class_accuracy=metrics.per_class_accuracy,
    class_counts=metrics.class_counts,
    title="Model Performance by Class",
    save_path="per_class_accuracy.png"
)
```

## Run Example Scripts

### Example 1: Basic Metrics Demo

```bash
cd libcll
python examples/error_asymmetry_example.py
```

This generates:
- `example_qd_map.png` - QD map visualization
- `example_qd_heatmap.png` - QD heatmap
- `example_balanced_per_class.png` - Balanced model analysis
- `example_biased_per_class.png` - Biased model analysis
- `example_evolution.png` - Metrics evolution over training

### Example 2: QD Integration with pyribs

```bash
cd libcll
python examples/qd_integration_example.py
```

This demonstrates full QD optimization with Error Asymmetry BDs.

## Key Metrics at a Glance

| Metric | Description | Range | Interpretation |
|--------|-------------|-------|----------------|
| **Overall Accuracy** | Mean accuracy across all samples | 0-1 | Higher is better |
| **Accuracy Variance** | Variance of per-class accuracies | 0-1 | **Primary BD**: Low = fair, High = biased |
| **Max Disparity** | Max(class_acc) - Min(class_acc) | 0-1 | **Alternative BD**: Class performance gap |
| **Per-Class Accuracy** | Accuracy for each class | 0-1 | Detailed class analysis |

## QD Map Interpretation

```
Accuracy Variance (BD)
        ↑
  0.10  |  Poor      Poor
        |  Specialist Specialist
  0.05  |  ─────────────────
        |  Balanced   Balanced
  0.00  |  Poor      IDEAL ✓
        |  Generalist Generalist
        └────────────────────→
        0.5  0.7  0.9  1.0
            Overall Accuracy
```

**Target Region**: Bottom-right (High accuracy + Low variance)

## Common Use Cases

### Use Case 1: Monitor Fairness During Training

```python
# Check if model is becoming biased
if metrics.accuracy_variance > 0.05:
    print("⚠️ Warning: Model showing bias toward certain classes")
    print(f"Best class: {metrics.max_accuracy:.2f}")
    print(f"Worst class: {metrics.min_accuracy:.2f}")
```

### Use Case 2: Select Fair Models

```python
# Choose between multiple trained models
models = [model1, model2, model3]
metrics_list = [eval_model(m) for m in models]

# Find the fairest high-performing model
fair_models = [
    (m, met) for m, met in zip(models, metrics_list)
    if met.overall_accuracy > 0.8 and met.accuracy_variance < 0.02
]

best_fair_model = max(fair_models, key=lambda x: x[1].overall_accuracy)
```

### Use Case 3: Identify Problem Classes

```python
# Find which classes need improvement
for i, acc in enumerate(metrics.per_class_accuracy):
    if acc < 0.5:
        print(f"Class {i} needs attention: {acc:.2%} accuracy")
```

## Integration with Existing Code

### Minimal Changes Required

If you're already using libcll strategies, **no code changes needed**! Error Asymmetry metrics are automatically computed.

### Manual Integration

If you want to compute metrics manually:

```python
# In your validation loop
predictions_list = []
targets_list = []

for batch in val_loader:
    x, y = batch
    pred = model(x).argmax(dim=1)
    predictions_list.append(pred)
    targets_list.append(y)

# Compute metrics
all_preds = torch.cat(predictions_list)
all_targets = torch.cat(targets_list)

from libcll.metrics import compute_error_asymmetry
metrics = compute_error_asymmetry(all_preds, all_targets, num_classes)

# Log to your preferred logger
logger.log("EA_Variance", metrics.accuracy_variance)
logger.log("EA_MaxDisparity", metrics.max_disparity)
```

## Testing

Run the test suite:

```bash
cd libcll
python tests/test_error_asymmetry.py
```

All tests should pass ✓

## Documentation

For detailed documentation, see:
- **Full Documentation**: `docs/ERROR_ASYMMETRY.md`
- **API Reference**: Module docstrings in `libcll/metrics/`
- **Examples**: `examples/error_asymmetry_example.py`

## Troubleshooting

### Issue: Import Error

```python
# Make sure libcll is in your Python path
import sys
sys.path.append("/path/to/libcll")

from libcll.metrics import compute_error_asymmetry
```

### Issue: Metrics Not Logged

Check that:
1. You're using a libcll Strategy (inherits from `Strategy` base class)
2. Validation is running (check `check_val_every_n_epoch`)
3. TensorBoard logger is configured

### Issue: Visualization Errors

Install visualization dependencies:
```bash
pip install matplotlib seaborn
```

## Next Steps

1. ✓ Run examples to understand the metrics
2. ✓ Integrate with your training pipeline
3. ✓ Monitor Error Asymmetry in TensorBoard
4. ✓ Use QD optimization for diverse model populations
5. ✓ Analyze per-class performance to improve weak classes

## Support

For issues or questions:
- Check the documentation: `docs/ERROR_ASYMMETRY.md`
- Review examples: `examples/`
- Run tests: `tests/test_error_asymmetry.py`

---

**Ready to start?** Run the examples:
```bash
cd libcll
python examples/error_asymmetry_example.py
```
