ff# Updated Error Asymmetry Implementation - CMA-QD Algorithms

## Major Changes

### 1. Fitness vs Behavior Descriptors Clarification

**IMPORTANT UPDATE**: Accuracy is now correctly used as FITNESS only, NOT as a behavior descriptor.

#### Previous (INCORRECT):
- BD1: Overall Accuracy ❌
- BD2: Accuracy Variance
- Problem: Mixes objective with behavior space

#### Current (CORRECT):
- **FITNESS**: Overall Accuracy (to MAXIMIZE) ✓
- **BD1**: Accuracy Variance (characteristic, no clear good/bad)
- **BD2**: Prediction Concentration (characteristic, no clear good/bad)
- **BD3-6**: Additional behavior descriptors

### 2. New Behavior Descriptors

All BDs are **characteristics without inherent "good" or "bad" direction**:

#### BD1: Accuracy Variance
- **What**: Spread of per-class accuracies
- **Why it's a BD**: Imbalanced data might benefit from specialization
- **Range**: 0.0 to 0.2+

#### BD2: Prediction Concentration
- **What**: How concentrated predictions are on certain classes
- **Why it's a BD**: Could match true imbalanced distribution or indicate bias
- **Range**: 0.0 to 1.0

#### BD3: Performance Entropy
- **What**: Entropy of per-class accuracy distribution
- **Why it's a BD**: High entropy = uniform, but uniformity isn't always best
- **Range**: 0.0 to log(num_classes)

#### BD4: Class Imbalance Adaptation
- **What**: Correlation between class frequency and accuracy
- **Why it's a BD**: Positive = favors frequent classes, Negative = favors rare classes
- **Range**: -1.0 to +1.0
- **Interpretation**:
  - `+1.0`: Perfect adaptation to frequent classes
  - `0.0`: No correlation with class frequency
  - `-1.0`: Specializes on rare classes

#### BD5: Specialization Index
- **What**: Coefficient of variation (std/mean) of accuracies
- **Why it's a BD**: High = specialist, Low = generalist - neither inherently better
- **Range**: 0.0 to 1.0+

#### BD6: Max Class Disparity  
- **What**: Performance gap between best and worst classes
- **Why it's a BD**: Large gaps aren't always bad if they match true difficulties
- **Range**: 0.0 to 1.0

### 3. CMA-Based QD Algorithms Support

Now supports three advanced QD algorithms:

#### CMA-ME (CMA-ES with MAP-Elites)
```python
from libcll.examples.qd_cma_algorithms import CMAQDConfig, run_qd_algorithm

config = CMAQDConfig(
    algorithm="cma_me",
    solution_dim=50,
    num_classes=10,
    grid_dims=(50, 50),
    num_emitters=15,
    batch_size=36,
    bd1_range=(0.0, 0.2),  # Variance
    bd2_range=(0.0, 1.0),  # Concentration
)

run_qd_algorithm(config)
```

**Features**:
- Two-stage improvement ranker
- Filter selection rule
- No-improvement restart rule
- Best for: Discovering diverse high-performing solutions

#### CMA-MEGA (CMA-ES with Gradient Arborescence)
```python
config = CMAQDConfig(
    algorithm="cma_mega",
    # ... other params
)

run_qd_algorithm(config)
```

**Features**:
- Differentiable QD (DQD)
- Uses gradient information
- Gradient arborescence emitter
- Best for: Fast convergence with gradient-based search

#### CMA-MAE (CMA-ES with Archive Learning Rate)
```python
config = CMAQDConfig(
    algorithm="cma_mae",
    learning_rate=0.01,  # Archive learning rate
    # ... other params
)

run_qd_algorithm(config)
```

**Features**:
- Archive with learning rate (adaptive thresholds)
- Improvement ranker
- Separable CMA-ES
- Best for: Continuous learning and adaptation

### 4. Updated QD Map Layout

**Correct QD Map Structure**:

```
       Y-axis: BD2 (e.g., Concentration)
         ↑
    1.0  |  [Solutions with different BDs]
         |     Each cell colored by FITNESS
    0.5  |       (brighter = higher accuracy)
         |
    0.0  |________________________________
         0.0              0.1          0.2
              X-axis: BD1 (e.g., Variance)

Objective (color gradient): Overall Accuracy
- Goal: Maximize accuracy (fitness)
- Map: Explore different behaviors (BDs)
```

**NOT like this** ❌:
```
Y: BD (Variance)
X: Accuracy  ← This puts fitness on axis!
```

### 5. Logged Metrics Update

#### TensorBoard Metrics

**Fitness (to maximize)**:
- `Valid_Fitness_Accuracy`
- `Test_Fitness_Accuracy`

**Behavior Descriptors** (no inherent good/bad):
- `Valid_BD_Variance`
- `Valid_BD_MaxDisparity`
- `Valid_BD_PerformanceEntropy`
- `Valid_BD_PredictionConcentration`
- `Valid_BD_ImbalanceAdaptation`
- `Valid_BD_SpecializationIndex`

**Auxiliary**:
- `Valid_EA_MinAccuracy`
- `Valid_EA_MaxAccuracy`
- `Test_EA_Class_{i}_Accuracy`

### 6. Example Usage

#### Basic Computation with New BDs

```python
from libcll.metrics import compute_error_asymmetry

metrics = compute_error_asymmetry(predictions, targets, num_classes)

# Fitness
print(f"Fitness (Accuracy): {metrics.overall_accuracy:.4f}")

# Behavior Descriptors (select any for QD map)
print(f"BD - Variance: {metrics.accuracy_variance:.6f}")
print(f"BD - Concentration: {metrics.prediction_concentration:.4f}")
print(f"BD - Entropy: {metrics.performance_entropy:.4f}")
print(f"BD - Imbalance Adapt: {metrics.class_imbalance_adaptation:+.4f}")
print(f"BD - Specialization: {metrics.specialization_index:.4f}")
```

#### QD Optimization with CMA-ME

```python
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from libcll.metrics import compute_error_asymmetry

# Create archive with 2 BDs
archive = GridArchive(
    solution_dim=model_params,
    dims=(50, 50),
    ranges=[
        (0.0, 0.2),  # BD1: Variance
        (0.0, 1.0),  # BD2: Concentration
    ],
)

# Create CMA-ME emitter
emitter = EvolutionStrategyEmitter(
    archive=archive,
    x0=initial_solution,
    sigma0=0.5,
    ranker="2imp",  # Two-stage improvement
    batch_size=36,
)

scheduler = Scheduler(archive=archive, emitters=[emitter])

# Optimization loop
for itr in range(1000):
    solutions = scheduler.ask()
    
    objectives = []
    measures = []
    
    for sol in solutions:
        # Train model with solution parameters
        model = train_model(sol)
        predictions = model.predict(test_data)
        
        # Compute metrics
        metrics = compute_error_asymmetry(predictions, targets, num_classes)
        
        # Fitness: overall accuracy (maximize)
        objectives.append(metrics.overall_accuracy)
        
        # BDs: characteristics (explore)
        measures.append([
            metrics.accuracy_variance,
            metrics.prediction_concentration
        ])
    
    scheduler.tell(objectives, measures)
```

#### Using Different BD Combinations

```python
# Option 1: Variance × Concentration
measures = [
    metrics.accuracy_variance,
    metrics.prediction_concentration
]

# Option 2: Specialization × Imbalance Adaptation
measures = [
    metrics.specialization_index,
    metrics.class_imbalance_adaptation
]

# Option 3: Entropy × Max Disparity
measures = [
    metrics.performance_entropy,
    metrics.max_disparity
]

# Option 4: Use CVTArchive for higher dimensions
from ribs.archives import CVTArchive

archive = CVTArchive(
    solution_dim=model_params,
    cells=10000,
    ranges=[
        (0.0, 0.2),   # Variance
        (0.0, 1.0),   # Concentration
        (0.0, 2.0),   # Entropy
        (-1.0, 1.0),  # Imbalance Adaptation
    ],
)

# Use 4D behavior space
measures = [
    metrics.accuracy_variance,
    metrics.prediction_concentration,
    metrics.performance_entropy,
    metrics.class_imbalance_adaptation
]
```

### 7. Why This Matters

#### Problem with Old Approach
Using accuracy as a BD meant the QD map was:
- X-axis: Accuracy (objective)
- Y-axis: Variance (BD)

This creates issues:
1. **Redundancy**: Optimizing accuracy while also using it as behavior space
2. **Poor exploration**: Archive cells at low accuracy never visited
3. **Conceptual confusion**: Mixes what we optimize with how it behaves

#### Benefits of New Approach
- X-axis: Variance (BD)
- Y-axis: Concentration (BD)
- Objective (color): Accuracy

This enables:
1. **True QD**: Explore diverse behaviors while maximizing fitness
2. **Better coverage**: All behavior combinations explored
3. **Useful diversity**: Find models with same accuracy but different characteristics

### 8. Interpretation Examples

#### Example 1: Imbalanced Dataset
```
Model A: accuracy=0.85, imbalance_adaptation=+0.8
→ Good fitness, favors frequent classes (might be optimal for imbalanced data)

Model B: accuracy=0.85, imbalance_adaptation=-0.8
→ Same fitness, favors rare classes (good for fairness)

Both models kept in archive - same fitness, different behaviors!
```

#### Example 2: Specialization Trade-offs
```
Model C: accuracy=0.88, specialization=0.1 (generalist)
→ Good at all classes fairly equally

Model D: accuracy=0.88, specialization=0.6 (specialist)
→ Excellent at some classes, poor at others

Archive keeps both - user can choose based on deployment needs!
```

## Running the New Examples

```bash
# Run CMA-based QD algorithms
cd libcll
python examples/qd_cma_algorithms.py

# This will run:
# 1. CMA-ME
# 2. CMA-MAE
# (Add CMA-MEGA in code if desired)

# Output:
# - qd_cma_output/cma_me/qd_heatmap.png
# - qd_cma_output/cma_me/metrics_evolution.png
# - qd_cma_output/cma_me/archive.csv
# - qd_cma_output/cma_mae/... (same structure)
```

## Summary of Changes

✅ **Fitness correctly separated from BDs**
✅ **6 behavior descriptors (all without clear good/bad)**
✅ **3 CMA-based QD algorithms supported**
✅ **Updated logging with clear BD_ and Fitness_ prefixes**
✅ **Comprehensive examples for all algorithms**
✅ **Proper QD map interpretation documentation**

## Migration Guide

If you were using the old version:

### Old Code:
```python
# This was mixing fitness and BD
measures = [accuracy, variance]  # WRONG!
```

### New Code:
```python
# Fitness separate from BDs
objective = metrics.overall_accuracy  # Fitness
measures = [
    metrics.accuracy_variance,         # BD1
    metrics.prediction_concentration   # BD2
]
```

### TensorBoard Logs:
```python
# Old: Valid_EA_Accuracy, Valid_EA_Variance
# New: Valid_Fitness_Accuracy, Valid_BD_Variance
```

This update makes the implementation theoretically correct and more powerful for QD optimization!
