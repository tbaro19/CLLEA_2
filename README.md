# Error Asymmetry: Behavior Descriptors for Quality Diversity

**New Feature!** `libcll` now includes **Error Asymmetry** metrics as Behavior Descriptors (BD) for Quality Diversity (QD) optimization. This feature helps identify and quantify model bias toward certain classes in continual learning.

## What is Error Asymmetry?

Error Asymmetry measures the performance disparity between classes, enabling:
- **Fair model selection**: Choose models with balanced class performance
- **QD optimization**: Maintain diverse model populations (generalists vs. specialists)
- **Model diagnosis**: Identify problematic classes needing attention

### Key Metrics

- **Accuracy Variance** (Primary BD): Measures fairness - low variance = generalist, high variance = specialist
- **Max Class Disparity** (Alternative BD): Performance gap between best and worst classes
- **Per-Class Accuracy**: Detailed analysis of class-specific performance

### Quick Start

Error Asymmetry metrics are **automatically tracked** during training:

```python
from libcll.strategies import SCL

# Train as usual - Error Asymmetry is logged automatically!
strategy = SCL(model=model, num_classes=10, lr=1e-4)
trainer = pl.Trainer(max_epochs=100)
trainer.fit(strategy, train_loader, val_loader)

# View in TensorBoard:
# - Valid_EA_Variance (primary BD)
# - Valid_EA_MaxDisparity (alternative BD)
# - Test_EA_Class_*_Accuracy (per-class)
```

### Manual Computation

```python
from libcll.metrics import compute_error_asymmetry

metrics = compute_error_asymmetry(predictions, targets, num_classes)
print(f"Accuracy: {metrics.overall_accuracy:.4f}")
print(f"Variance (BD): {metrics.accuracy_variance:.6f}")
print(f"Interpretation: {'Generalist' if metrics.accuracy_variance < 0.01 else 'Specialist'}")
```

### QD Map Visualization

```python
from libcll.metrics.visualize import plot_qd_map_2d

# Create QD map: [Overall Accuracy] vs [Accuracy Variance]
plot_qd_map_2d(
    accuracy=accuracies,
    behavior_descriptor=variances,
    xlabel="Overall Accuracy",
    ylabel="Accuracy Variance (BD)",
    save_path="qd_map.png"
)
```

### Integration with pyribs QD Framework

```python
from ribs.archives import GridArchive
from libcll.metrics import compute_error_asymmetry

# Create QD archive
archive = GridArchive(
    solution_dim=model_params,
    dims=(50, 50),
    ranges=[(0.0, 1.0), (0.0, 0.2)],  # [accuracy, variance]
)

# Add solutions with Error Asymmetry BDs
metrics = compute_error_asymmetry(predictions, targets, num_classes)
archive.add(
    solution=model_weights,
    objective=metrics.overall_accuracy,
    measures=[metrics.overall_accuracy, metrics.accuracy_variance]
)
```

### Documentation

- **Quick Start**: `docs/QUICKSTART_ERROR_ASYMMETRY.md`
- **Full Guide**: `docs/ERROR_ASYMMETRY.md`
- **Architecture**: `ARCHITECTURE.md`
- **Examples**: `examples/error_asymmetry_example.py`, `examples/qd_integration_example.py`

### Run Examples

```bash
# Basic usage examples
python examples/error_asymmetry_example.py

# QD optimization with pyribs
python examples/qd_integration_example.py
```

# Documentation

The documentation for the latest release is available on [readthedocs](https://libcll.readthedocs.io/en/latest/). Feedback, questions, and suggestions are highly encouraged. Contributions to improve the documentation are warmly welcomed and greatly appreciated!

