# Error Asymmetry BD Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        libcll Framework                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐         ┌──────────────┐                      │
│  │   Training   │────────▶│  Validation  │                      │
│  │     Loop     │         │     Step     │                      │
│  └──────────────┘         └──────┬───────┘                      │
│                                   │                              │
│                                   │ Store predictions/targets    │
│                                   ▼                              │
│                          ┌────────────────┐                      │
│                          │  Epoch End     │                      │
│                          │  Callback      │                      │
│                          └────────┬───────┘                      │
│                                   │                              │
│                                   │ Compute metrics              │
│                                   ▼                              │
│                    ┌──────────────────────────┐                 │
│                    │  Error Asymmetry Module  │                 │
│                    │  (libcll.metrics)        │                 │
│                    ├──────────────────────────┤                 │
│                    │ • compute_error_asymmetry│                 │
│                    │ • compute_per_class_acc  │                 │
│                    │ • compute_variance       │                 │
│                    │ • compute_disparity      │                 │
│                    └──────────┬───────────────┘                 │
│                               │                                  │
│                               │ Return metrics                   │
│                               ▼                                  │
│              ┌────────────────────────────────┐                 │
│              │   ErrorAsymmetryMetrics        │                 │
│              ├────────────────────────────────┤                 │
│              │ • overall_accuracy             │                 │
│              │ • per_class_accuracy           │                 │
│              │ • accuracy_variance (BD) ◀─────┼──── Primary BD  │
│              │ • max_disparity (BD) ◀─────────┼──── Alt BD      │
│              │ • min/max_accuracy             │                 │
│              └────────────┬───────────────────┘                 │
│                           │                                      │
│                           │ Log to                               │
│                           ▼                                      │
│                  ┌─────────────────┐                            │
│                  │   TensorBoard   │                            │
│                  │     Logger      │                            │
│                  └─────────────────┘                            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

                              │
                              │ Use for QD
                              ▼

┌─────────────────────────────────────────────────────────────────┐
│                    pyribs QD Framework                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────┐           │
│  │              GridArchive / CVTArchive            │           │
│  │                                                  │           │
│  │  Dimensions:                                     │           │
│  │    X-axis: Overall Accuracy (Objective)         │           │
│  │    Y-axis: Accuracy Variance (BD)               │           │
│  │                                                  │           │
│  │  ┌────────────────────────────────────┐        │           │
│  │  │         QD Map (50x50 grid)        │        │           │
│  │  │                                    │        │           │
│  │  │  Var    ●   ●     ← Specialists   │        │           │
│  │  │  0.10   ● ● ● ●                   │        │           │
│  │  │         ● ● ● ● ●                 │        │           │
│  │  │  0.05   ● ● ● ● ●                 │        │           │
│  │  │         ● ● ● ● ● ●               │        │           │
│  │  │  0.00 ──●─●─●─●─●─●→ Accuracy     │        │           │
│  │  │        0.5   0.7  0.9   1.0       │        │           │
│  │  │                                    │        │           │
│  │  │  Target: Bottom-right (✓)         │        │           │
│  │  └────────────────────────────────────┘        │           │
│  └──────────────────────────────────────────────────┘           │
│                           │                                      │
│                           │ Managed by                           │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────┐               │
│  │            Scheduler + Emitters              │               │
│  │                                              │               │
│  │  • EvolutionStrategyEmitter                 │               │
│  │  • GaussianEmitter                          │               │
│  │  • GradientArborescenceEmitter              │               │
│  └──────────────────────────────────────────────┘               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Training Data
     │
     ▼
┌─────────┐
│  Model  │
└────┬────┘
     │
     ▼
Predictions ──────┐
                  │
Targets ──────────┤
                  │
                  ▼
        ┌─────────────────────┐
        │  Error Asymmetry    │
        │  Computation        │
        └─────────┬───────────┘
                  │
         ┌────────┴────────┐
         │                 │
         ▼                 ▼
    ┌─────────┐      ┌─────────┐
    │ Logging │      │   QD    │
    │(TBoard) │      │ Archive │
    └─────────┘      └─────────┘
```

## Component Relationships

```
┌──────────────────────────────────────────────────────────┐
│                     User's Code                          │
│  ┌────────────────────────────────────────────────┐     │
│  │  from libcll.strategies import Strategy        │     │
│  │  from libcll.metrics import compute_ea         │     │
│  │  from libcll.metrics.visualize import plot_qd  │     │
│  └────────────────────────────────────────────────┘     │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│                  libcll.metrics                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │              error_asymmetry.py                  │    │
│  │  • compute_error_asymmetry()                    │    │
│  │  • compute_per_class_accuracy()                 │    │
│  │  • compute_accuracy_variance()                  │    │
│  │  • compute_class_accuracy_disparity()           │    │
│  │  • ErrorAsymmetryMetrics (NamedTuple)           │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │               visualize.py                       │    │
│  │  • plot_qd_map_2d()                             │    │
│  │  • plot_qd_heatmap()                            │    │
│  │  • plot_per_class_accuracy_bars()               │    │
│  │  • plot_error_asymmetry_evolution()             │    │
│  │  • create_qd_report()                           │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────┬───────────────────────────────────────────┘
               │
               │ Used by
               ▼
┌──────────────────────────────────────────────────────────┐
│              libcll.strategies.Strategy                  │
│  ┌─────────────────────────────────────────────────┐    │
│  │  • validation_step()                            │    │
│  │    - Store predictions/targets                  │    │
│  │  • on_validation_epoch_end()                    │    │
│  │    - Compute Error Asymmetry                    │    │
│  │    - Log to TensorBoard                         │    │
│  │  • test_step()                                  │    │
│  │    - Store predictions/targets                  │    │
│  │  • on_test_epoch_end()                          │    │
│  │    - Compute Error Asymmetry                    │    │
│  │    - Log per-class accuracies                   │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

## Metric Computation Pipeline

```
Input: predictions, targets, num_classes
  │
  ├──▶ Overall Accuracy
  │      └──▶ mean(predictions == targets)
  │
  ├──▶ Per-Class Accuracy
  │      └──▶ For each class i:
  │            acc[i] = correct[i] / total[i]
  │
  ├──▶ Accuracy Variance (Primary BD)
  │      └──▶ var(per_class_accuracy)
  │
  ├──▶ Accuracy Std Dev
  │      └──▶ std(per_class_accuracy)
  │
  ├──▶ Max Disparity (Alternative BD)
  │      └──▶ max(per_class_accuracy) - min(per_class_accuracy)
  │
  ├──▶ Min/Max Class Accuracy
  │      └──▶ min(per_class_accuracy), max(per_class_accuracy)
  │
  └──▶ Class Counts
         └──▶ Number of samples per class

Output: ErrorAsymmetryMetrics (NamedTuple)
```

## QD Integration Flow

```
┌─────────────────────────────────────────────────────────┐
│              QD Optimization Loop                        │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  1. Ask Scheduler   │
│     for solutions   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Evaluate each   │
│     solution        │
│     (train model)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Compute EA      │
│     metrics         │
│  • accuracy (obj)   │
│  • variance (BD)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. Tell Scheduler  │
│     objective=acc   │
│     measures=[acc,  │
│              var]   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. Archive adds    │
│     elite solutions │
│     to QD map       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  6. Visualize QD    │
│     map showing     │
│     generalists vs  │
│     specialists     │
└─────────────────────┘
```

## File Structure

```
libcll/
├── libcll/
│   ├── metrics/
│   │   ├── __init__.py              ← Module exports
│   │   ├── error_asymmetry.py       ← Core computation (370 lines)
│   │   └── visualize.py             ← QD visualizations (470 lines)
│   └── strategies/
│       └── Strategy.py              ← Modified (auto-tracking)
├── docs/
│   ├── ERROR_ASYMMETRY.md           ← Full documentation
│   └── QUICKSTART_ERROR_ASYMMETRY.md ← Quick start guide
├── examples/
│   ├── error_asymmetry_example.py   ← Usage examples
│   └── qd_integration_example.py    ← pyribs integration
├── tests/
│   └── test_error_asymmetry.py      ← Test suite (11 tests)
└── IMPLEMENTATION_SUMMARY.md        ← This summary
```

## Key Classes and Functions

```
ErrorAsymmetryMetrics (NamedTuple)
├── overall_accuracy: float
├── per_class_accuracy: np.ndarray
├── accuracy_variance: float          ◀── Primary BD
├── accuracy_std: float
├── max_disparity: float              ◀── Alternative BD
├── min_accuracy: float
├── max_accuracy: float
├── class_counts: np.ndarray
└── num_classes: int

compute_error_asymmetry(predictions, targets, num_classes)
├── compute_per_class_accuracy()
├── compute_accuracy_variance()
└── compute_class_accuracy_disparity()

Visualization Functions
├── plot_qd_map_2d()                  ◀── Scatter plot
├── plot_qd_heatmap()                 ◀── Archive heatmap
├── plot_per_class_accuracy_bars()    ◀── Per-class analysis
├── plot_error_asymmetry_evolution()  ◀── Training progress
└── create_qd_report()                ◀── Comprehensive report
```

## Integration Points

```
User's Training Code
        │
        ▼
┌───────────────────┐
│ PyTorch Lightning │
│     Trainer       │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  libcll Strategy  │◀──── Inherits from Strategy base class
│   (e.g., SCL)     │
└────────┬──────────┘
         │
         │ Automatic
         │ integration
         ▼
┌───────────────────┐
│  Error Asymmetry  │
│     Tracking      │
└────────┬──────────┘
         │
         ├──▶ TensorBoard (logging)
         └──▶ pyribs Archive (QD)
```

## Logged Metrics Hierarchy

```
TensorBoard Logs
├── Training
│   └── Train_Loss
├── Validation
│   ├── Valid_URE / Valid_SCEL / Valid_Accuracy
│   ├── Valid_EA_Accuracy
│   ├── Valid_EA_Variance          ◀── Primary BD
│   ├── Valid_EA_StdDev
│   ├── Valid_EA_MaxDisparity      ◀── Alternative BD
│   ├── Valid_EA_MinAccuracy
│   └── Valid_EA_MaxAccuracy
└── Testing
    ├── Test_Accuracy
    ├── Test_EA_Accuracy
    ├── Test_EA_Variance
    ├── Test_EA_MaxDisparity
    ├── Test_EA_Class_0_Accuracy
    ├── Test_EA_Class_1_Accuracy
    └── ...
```

## Workflow Summary

1. **Training**: Model trains normally with libcll strategies
2. **Validation**: Error Asymmetry automatically computed each epoch
3. **Logging**: Metrics logged to TensorBoard
4. **Analysis**: View metrics to identify generalist vs specialist behavior
5. **QD (Optional)**: Use metrics as BDs in pyribs QD optimization
6. **Visualization**: Generate QD maps and reports

## Quick Reference

| Metric | Type | Purpose | Threshold |
|--------|------|---------|-----------|
| Overall Accuracy | Objective | Model performance | Maximize |
| Accuracy Variance | BD | Class fairness | < 0.01: Fair |
| Max Disparity | BD | Performance gap | < 0.2: Balanced |
| Per-Class Acc | Analysis | Find weak classes | - |

---

This architecture enables automatic tracking of model fairness across classes
during continual learning, with seamless integration into both training
workflows and Quality Diversity optimization.
