# Error Asymmetry: Behavior Descriptors for Continual Learning

This module provides **Error Asymmetry** metrics as Behavior Descriptors (BD) for Quality Diversity (QD) optimization in Continual Learning with Label Errors (CLL).

## Overview

In CLL, one major challenge is that models may learn to become **biased toward certain classes**. This bias can lead to:
- High accuracy on some classes
- Poor accuracy on others
- Unfair model behavior

Error Asymmetry metrics help identify and quantify this behavior, enabling:
1. **Fair model selection**: Choose models that balance performance across classes
2. **Specialist vs. Generalist identification**: Understand model behavior patterns
3. **Quality Diversity optimization**: Maintain diverse model populations in QD archives

## Key Concepts

### Fitness vs Behavior Descriptors

**IMPORTANT DISTINCTION:**
- **FITNESS**: Overall Accuracy - This is what we want to MAXIMIZE
- **BEHAVIOR DESCRIPTORS (BDs)**: Characteristics with NO clear "good" or "bad" direction

### Behavior Descriptors (BDs)

#### 1. Accuracy Variance
**Definition**: Variance of per-class accuracies
```python
variance = Var([Acc_class_0, Acc_class_1, ..., Acc_class_N])
```
**Interpretation**: High variance = specialist, Low variance = generalist
**Why it's a BD**: Depends on data distribution - imbalanced data might benefit from some specialization

#### 2. Max Class Disparity
**Definition**: Maximum difference between class accuracies
```python
max_disparity = max(Acc_i) - min(Acc_j)
```
**Why it's a BD**: Large gaps aren't always bad if they match true class difficulties

#### 3. Performance Entropy
**Definition**: Entropy of per-class accuracy distribution
```python
entropy = -sum(p_i * log(p_i))  where p_i = Acc_i / sum(Acc)
```
**Why it's a BD**: Higher entropy = more uniform, but uniformity isn't always optimal

#### 4. Prediction Concentration
**Definition**: How concentrated predictions are on certain classes
```python
concentration = Var(prediction_distribution) * num_classes
```
**Why it's a BD**: High concentration could match imbalanced data or indicate bias

#### 5. Class Imbalance Adaptation
**Definition**: Correlation between class frequency and accuracy
```python
adaptation = Corr(class_counts, class_accuracies)
```
**Why it's a BD**: Positive = favors frequent classes, Negative = favors rare classes
**Values**: -1 (rare-class specialist) to +1 (frequent-class specialist)

#### 6. Specialization Index
**Definition**: Coefficient of variation of accuracies
```python
specialization = Std(accuracies) / Mean(accuracies)
```
**Why it's a BD**: High = specialist, Low = generalist - neither is inherently better

## Quality Diversity Map Interpretation

### QD Map Layout (Correct Usage)

```
       BD2: Prediction Concentration
         ↑
    1.0  |  ●  ●  ●    ← Different prediction patterns
         |  ●  ●  ●  ●
    0.5  |  ●  ●  ●  ●  ●
         |  ●  ●  ●  ●  ●  ●
    0.0  |__●__●__●__●__●__●_→  BD1: Accuracy Variance
         0.0      0.1       0.2

Fitness (color): Overall Accuracy (to maximize)
- Brighter colors = Higher accuracy (better)
- Darker colors = Lower accuracy (worse)

BDs (axes): Characteristics (no inherent good/bad)
- Different positions = Different behaviors
- All positions potentially useful depending on context
```

### Why BDs are NOT on Accuracy Axis

**INCORRECT** ❌:
```
Y: Variance (BD)
X: Accuracy (FITNESS)  ← This mixes fitness with BD!
```

**CORRECT** ✅:
```
Y: BD2 (e.g., Concentration)
X: BD1 (e.g., Variance)
Color/Objective: Accuracy (FITNESS)
```

### Multi-Dimensional BD Space

You can use different combinations:

**2D Examples**:
1. Variance × Concentration
2. Variance × Imbalance Adaptation
3. Specialization × Entropy
4. Concentration × Imbalance Adaptation

**3D+ Examples** (with pyribs CVTArchive):
- Variance × Concentration × Entropy
- All 6 BDs in a 6D behavior space

## Usage Examples

### 1. Basic Computation

```python
import torch
from libcll.metrics import compute_error_asymmetry

# Your model predictions and targets
predictions = torch.tensor([0, 1, 1, 2, 0, 1, 2, 2])
targets = torch.tensor([0, 1, 2, 2, 0, 1, 2, 2])
num_classes = 3

# Compute error asymmetry metrics
metrics = compute_error_asymmetry(predictions, targets, num_classes)

print(f"Overall Accuracy: {metrics.overall_accuracy:.4f}")
print(f"Accuracy Variance (BD): {metrics.accuracy_variance:.6f}")
print(f"Max Disparity: {metrics.max_disparity:.4f}")
```

### 2. Integration with PyTorch Lightning

The Strategy base class now automatically tracks error asymmetry metrics:

```python
from libcll.strategies import Strategy

class MyStrategy(Strategy):
    def training_step(self, batch, batch_idx):
        # Your training logic
        pass
    
    # Error asymmetry is automatically computed in validation_step
    # and logged to TensorBoard
```

**Logged Metrics**:
- `Valid_EA_Accuracy`: Overall validation accuracy
- `Valid_EA_Variance`: Accuracy variance (primary BD)
- `Valid_EA_MaxDisparity`: Max class disparity
- `Valid_EA_MinAccuracy`: Worst-performing class
- `Valid_EA_MaxAccuracy`: Best-performing class
- `Test_EA_Class_{i}_Accuracy`: Per-class test accuracies

### 3. Visualization

```python
from libcll.metrics.visualize import plot_qd_map_2d, plot_per_class_accuracy_bars

# Create QD map
fig = plot_qd_map_2d(
    accuracy=overall_accuracies,
    behavior_descriptor=variances,
    xlabel="Overall Accuracy",
    ylabel="Accuracy Variance (BD)",
    save_path="qd_map.png"
)

# Plot per-class performance
fig = plot_per_class_accuracy_bars(
    per_class_accuracy=metrics.per_class_accuracy,
    class_counts=metrics.class_counts,
    save_path="per_class.png"
)
```

### 4. Creating Comprehensive Reports

```python
from libcll.metrics.visualize import create_qd_report

metrics_dict = {
    'accuracy': 0.85,
    'variance': 0.042,
    'max_disparity': 0.35,
    'min_accuracy': 0.68,
    'max_accuracy': 0.96,
}

create_qd_report(
    metrics_dict=metrics_dict,
    per_class_accuracy=metrics.per_class_accuracy,
    class_counts=metrics.class_counts,
    output_dir="reports",
    prefix="my_model"
)
```

## Integration with pyribs QD Framework

Error Asymmetry metrics can be used with the pyribs library for Quality Diversity optimization:

```python
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from libcll.metrics import compute_error_asymmetry

# Create QD archive with 2D behavior space
archive = GridArchive(
    solution_dim=model_parameters,
    dims=(50, 50),  # 50x50 grid
    ranges=[(0.0, 1.0), (0.0, 0.2)],  # [accuracy, variance]
)

# During evaluation
predictions = model.predict(test_data)
metrics = compute_error_asymmetry(predictions, targets, num_classes)

# Add to archive
archive.add(
    solution=model.parameters(),
    objective=metrics.overall_accuracy,  # Maximize accuracy
    measures=[metrics.overall_accuracy, metrics.accuracy_variance]  # BDs
)
```

## Use Cases

### 1. Fair Model Selection
Select models that balance accuracy across all classes, avoiding bias.

### 2. Diverse Model Populations
Maintain a diverse set of models with different specializations.

### 3. Curriculum Learning
Track how class-specific performance evolves during training.

### 4. Model Diagnosis
Identify which classes are problematic and require attention.

### 5. Multi-Objective Optimization
Balance overall accuracy with fairness (low variance).

## Files

- `libcll/metrics/error_asymmetry.py` - Core metric computation
- `libcll/metrics/visualize.py` - Visualization utilities
- `libcll/metrics/__init__.py` - Module exports
- `libcll/strategies/Strategy.py` - Automatic integration with training
- `examples/error_asymmetry_example.py` - Usage examples

## Running Examples

```bash
# Run example script
cd libcll
python examples/error_asymmetry_example.py

# This will generate:
# - example_qd_map.png
# - example_qd_heatmap.png
# - example_balanced_per_class.png
# - example_biased_per_class.png
# - example_evolution.png
# - qd_reports/
```

## References

- **Quality Diversity**: Fontaine et al. "Illuminating search spaces by mapping elites" (2015)
- **CMA-ME**: Fontaine & Nikolaidis "Differentiable Quality Diversity" (2023)
- **pyribs**: https://github.com/icaros-usc/pyribs

## Contributing

To extend the Error Asymmetry metrics:

1. Add new metrics to `error_asymmetry.py`
2. Add visualization functions to `visualize.py`
3. Update the `ErrorAsymmetryMetrics` named tuple
4. Add logging to `Strategy.py` if needed

## License

Same as libcll project license.
