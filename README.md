# CLLEA: Continual Learning with Label Error Asymmetry

A comprehensive framework for Continual Learning with Label Errors (CLLEA) that integrates Quality Diversity (QD) optimization and Error Asymmetry metrics to train fair and diverse neural networks.

## 🎯 Overview

CLLEA addresses a critical challenge in continual learning: **label noise and class bias**. Traditional approaches often produce models that become specialists (excellent at some classes, poor at others) or suffer from unfair performance across different classes.

Our framework provides:
- **Error Asymmetry Metrics**: Quantify model bias and fairness
- **Quality Diversity Optimization**: Maintain diverse populations of models with different specializations
- **Real Training Integration**: Train actual neural networks (not simulations) with PyTorch Lightning
- **Comprehensive Visualization**: Interactive QD maps, heatmaps, and performance analysis

## 🚀 Key Features

### 1. Error Asymmetry as Behavior Descriptors
- **Accuracy Variance**: Measures specialization vs generalization
- **Max Class Disparity**: Quantifies largest performance gaps
- **Prediction Concentration**: Tracks model bias toward certain classes
- **Performance Entropy**: Measures uniformity of class performance
- **Class Imbalance Adaptation**: Correlation between class frequency and accuracy
- **Specialization Index**: Coefficient of variation across classes

### 2. Quality Diversity Optimization
- Integration with [pyribs](https://github.com/icaros-usc/pyribs) QD framework
- Support for CMA-ES, CMA-ME, MAP-Elites algorithms
- 2D and multi-dimensional behavior spaces
- Archive-based diversity maintenance

### 3. Real Neural Network Training
- PyTorch Lightning integration
- Multiple architectures: Linear, MLP, ResNet18/34, DenseNet
- Various continual learning strategies: SCL, URE, FWD, DM, CPE, PC
- MNIST, CIFAR-10 support with label noise injection

### 4. Comprehensive Analysis Tools
- Interactive QD maps and heatmaps
- Per-class performance analysis
- Evolution tracking over training iterations
- Model comparison and selection utilities

## 📁 Project Structure

```
CLLEA_2/
├── libcll/                          # Core CLLEA framework
│   ├── datasets/                    # Dataset loaders with noise injection
│   ├── models/                      # Neural network architectures
│   ├── strategies/                  # Continual learning strategies
│   ├── metrics/                     # Error asymmetry computation
│   │   ├── error_asymmetry.py      # Core metrics implementation
│   │   └── visualize.py            # Visualization utilities
│   ├── examples/                    # Usage examples and tutorials
│   │   ├── error_asymmetry_example.py     # Basic usage examples
│   │   ├── qd_integration_example.py      # QD optimization demo
│   │   ├── qd_cma_algorithms.py           # CMA-ES/CMA-ME examples  
│   │   ├── qd_real_training.py            # Real model training
│   │   └── qd_simple_cpu.py               # CPU-only training
│   └── docs/                        # Documentation
│       ├── ERROR_ASYMMETRY.md      # Detailed error asymmetry guide
│       └── QUICKSTART_ERROR_ASYMMETRY.md
└── pyribs-master/                   # QD optimization framework
    └── ribs/                        # Archives, emitters, schedulers
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+ (CPU or CUDA)
- CUDA 11.0+ (optional, for GPU training)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd CLLEA_2

# Install libcll framework
cd libcll
pip install -e .

# Install pyribs for QD optimization  
cd ../pyribs-master
pip install -e .

# Install additional dependencies
pip install pandas matplotlib seaborn plotly
```

## 🚀 Quick Start

### 1. Basic Error Asymmetry Analysis

```python
from libcll.metrics import compute_error_asymmetry
import torch

# Your model predictions and ground truth
predictions = torch.tensor([0, 1, 1, 2, 0, 1, 2, 2])
targets = torch.tensor([0, 1, 2, 2, 0, 1, 2, 2])

# Compute error asymmetry metrics
metrics = compute_error_asymmetry(predictions, targets, num_classes=3)

print(f"Overall Accuracy: {metrics.overall_accuracy:.4f}")
print(f"Accuracy Variance (BD1): {metrics.accuracy_variance:.6f}")  
print(f"Max Class Disparity (BD2): {metrics.max_disparity:.4f}")
print(f"Per-class Accuracies: {metrics.per_class_accuracy}")
```

### 2. Run Example Demonstrations

```bash
cd libcll

# Basic error asymmetry examples and visualizations
python examples/error_asymmetry_example.py

# QD optimization with simulated models (fast)
python examples/qd_integration_example.py

# Real model training with QD optimization (slow but real)
python examples/qd_simple_cpu.py
```

### 3. Train Real Models with QD Optimization

```bash
# Train 8 different model configurations  
python examples/qd_real_training.py --dataset mnist --n_models 8

# Or use CPU-only version (no CUDA required)
python examples/qd_simple_cpu.py
```

## 📊 Key Concepts

### Fitness vs Behavior Descriptors

**CRITICAL DISTINCTION:**
- **FITNESS (Objective)**: Overall Accuracy - what we want to MAXIMIZE
- **BEHAVIOR DESCRIPTORS**: Model characteristics with NO inherent "good/bad" direction

### Behavior Descriptors (BDs)

#### 1. Accuracy Variance
```python
variance = Var([Acc_class_0, Acc_class_1, ..., Acc_class_N])
```
- **High variance** = specialist model
- **Low variance** = generalist model
- **Why it's a BD**: Depends on data distribution - imbalanced data might benefit from some specialization

#### 2. Max Class Disparity
```python
max_disparity = max(Acc_i) - min(Acc_j)
```
- **Large gaps** aren't always bad if they match true class difficulties
- **Small gaps** indicate balanced performance

#### 3. Performance Entropy
```python
entropy = -sum(p_i * log(p_i))  where p_i = Acc_i / sum(Acc)
```
- **Higher entropy** = more uniform performance
- **Lower entropy** = concentrated performance

#### 4. Prediction Concentration
```python
concentration = Var(prediction_distribution) * num_classes
```
- **High concentration** could match imbalanced data or indicate bias
- **Low concentration** indicates balanced predictions

#### 5. Class Imbalance Adaptation
```python
adaptation = Corr(class_counts, class_accuracies)
```
- **Positive** = favors frequent classes
- **Negative** = favors rare classes
- **Values**: -1 (rare-class specialist) to +1 (frequent-class specialist)

#### 6. Specialization Index
```python
specialization = Std(accuracies) / Mean(accuracies)
```
- **High** = specialist model
- **Low** = generalist model

### Quality Diversity Map Layout

```
       BD2: Max Class Disparity
         ↑
    1.0  |  ●  ●  ●    ← Larger gaps between classes  
         |  ●  ●  ●  ●
    0.5  |  ●  ●  ●  ●  ●
         |  ●  ●  ●  ●  ●  ●
    0.0  |__●__●__●__●__●__●_→  BD1: Accuracy Variance
         0.0      0.1       0.2

Color: Overall Accuracy (fitness to maximize)
X-axis: Accuracy Variance (BD1) - specialization measure
Y-axis: Max Class Disparity (BD2) - largest performance gap
```

### Model Behavior Types

- **Generalists**: Low variance, low disparity (consistent across classes)
- **Specialists**: High variance, high disparity (excellent at few classes)  
- **Balanced**: Medium variance, low disparity (good overall performance)
- **Biased**: Low variance, high disparity (systematic class preference)

## 🧪 Examples and Use Cases

### 1. Model Selection for Fairness

```python
# Load QD archive results
import pandas as pd
df = pd.read_csv('qd_output/archive.csv')

# Find fair models (low disparity, high accuracy)
fair_models = df[(df['measures_1'] < 0.1) & (df['objective'] > 0.85)]
print(f"Found {len(fair_models)} fair models")
```

### 2. Integration with PyTorch Lightning

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

### 3. Integration with pyribs QD Framework

```python
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from libcll.metrics import compute_error_asymmetry

# Create QD archive with 2D behavior space
archive = GridArchive(
    solution_dim=model_parameters,
    dims=(50, 50),  # 50x50 grid
    ranges=[(0.0, 0.2), (0.0, 1.0)],  # [variance, max_disparity]
)

# During evaluation
predictions = model.predict(test_data)
metrics = compute_error_asymmetry(predictions, targets, num_classes)

# Add to archive
archive.add(
    solution=model.parameters(),
    objective=metrics.overall_accuracy,  # Maximize accuracy (FITNESS)
    measures=[metrics.accuracy_variance, metrics.max_disparity]  # BDs
)
```

### 4. Visualization

```python
from libcll.metrics.visualize import plot_qd_map_2d, plot_per_class_accuracy_bars

# Create QD map
fig = plot_qd_map_2d(
    accuracy=variances,
    behavior_descriptor=disparities,
    values=accuracies,  # Color by accuracy
    xlabel="Accuracy Variance (BD1)",
    ylabel="Max Class Disparity (BD2)",
    title="QD Map: Model Behavior Space",
    save_path="qd_map.png"
)

# Plot per-class performance
fig = plot_per_class_accuracy_bars(
    per_class_accuracy=metrics.per_class_accuracy,
    class_counts=metrics.class_counts,
    save_path="per_class.png"
)
```

### 5. Comprehensive QD Report

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

### 6. Multi-Architecture Comparison

```python
# Compare different architectures in QD space
results = {}
for arch in ['Linear', 'MLP', 'ResNet18']:
    results[arch] = train_with_qd(architecture=arch)
    
plot_architecture_comparison(results, save_path="arch_comparison.png")
```

## 📈 Output Files and Visualizations

### Generated Files
- `qd_output/archive.csv` - Complete QD archive results
- `qd_output/metrics_history.json` - Evolution over iterations
- `qd_map.png` - 2D behavior space visualization  
- `qd_heatmap.png` - Density heatmap of solutions
- `per_class_analysis.png` - Class-specific performance breakdown
- `evolution.png` - Training progress over time
- `qd_reports/` - Comprehensive analysis reports

### Interactive Analysis
- Plotly-based interactive QD maps
- Hover tooltips with model configurations
- Zoomable behavior space exploration
- Model parameter inspection

## 🔧 Advanced Usage

### Custom Behavior Descriptors

```python
# Define custom BD combinations
behavior_descriptors = [
    'accuracy_variance',         # Specialization measure
    'prediction_concentration',  # Bias toward certain classes  
    'performance_entropy',       # Uniformity measure
]

# Create 3D QD archive
from ribs.archives import CVTArchive
archive = CVTArchive(
    solution_dim=model_params,
    cells=1000,
    ranges=[(0.0, 0.2), (0.0, 1.0), (0.0, 3.0)]
)
```

### Multi-Dimensional BD Space

**2D Examples**:
1. Variance × Max Disparity (default in examples)
2. Variance × Concentration
3. Variance × Imbalance Adaptation
4. Specialization × Entropy
5. Concentration × Imbalance Adaptation

**3D+ Examples** (with pyribs CVTArchive):
- Variance × Concentration × Entropy
- All 6 BDs in a 6D behavior space

### Integration with Existing Training

```python
from libcll.strategies import SCL
from libcll.metrics import compute_error_asymmetry

# Your existing training loop
for epoch in range(num_epochs):
    # ... training code ...
    
    # Add error asymmetry tracking
    with torch.no_grad():
        predictions = model(test_data)
        metrics = compute_error_asymmetry(predictions, targets, num_classes)
        
        # Log or store metrics
        logger.log_metrics(metrics._asdict(), step=epoch)
```

## 📚 Available Examples

### Basic Examples
```bash
# Error asymmetry computation and visualization
python examples/error_asymmetry_example.py

# QD optimization with simulated evaluation (fast)
python examples/qd_integration_example.py

# CMA-ES and CMA-ME algorithms
python examples/qd_cma_algorithms.py
```

### Real Training Examples
```bash
# CPU-only real training (works without CUDA)
python examples/qd_simple_cpu.py

# Full real training with GPU support
python examples/qd_real_training.py --dataset mnist --n_models 15
```

## 🎯 Use Cases

### 1. Fair Model Selection
Select models that balance accuracy across all classes, avoiding bias toward specific classes.

### 2. Diverse Model Populations
Maintain a diverse set of models with different specializations for ensemble learning or model selection.

### 3. Curriculum Learning Analysis
Track how class-specific performance evolves during training to design better curricula.

### 4. Model Diagnosis
Identify which classes are problematic and require attention or additional training data.

### 5. Multi-Objective Optimization
Balance overall accuracy with fairness metrics (low variance, low disparity).

### 6. Continual Learning Strategy Comparison
Compare how different continual learning strategies affect class-wise performance.

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Bug Reports**: Use GitHub issues with detailed reproduction steps
2. **Feature Requests**: Propose new behavior descriptors or QD algorithms  
3. **Pull Requests**: Follow our coding standards and include tests
4. **Documentation**: Help improve examples and tutorials

### Development Setup

```bash
# Clone with development dependencies
git clone <repo> && cd CLLEA_2
pip install -e "libcll[dev]"
pip install -e "pyribs-master[dev]"

# Run tests
pytest libcll/tests/
pytest pyribs-master/tests/

# Code formatting  
pre-commit install
black libcll/ pyribs-master/
```

### Extending Error Asymmetry Metrics

To add new metrics:

1. Add new metrics to `libcll/metrics/error_asymmetry.py`
2. Add visualization functions to `libcll/metrics/visualize.py`
3. Update the `ErrorAsymmetryMetrics` named tuple
4. Add logging to `libcll/strategies/Strategy.py` if needed
5. Update documentation and examples

## 📚 Documentation

- **[Error Asymmetry Guide](libcll/docs/ERROR_ASYMMETRY.md)**: Detailed behavior descriptor explanations
- **[Quick Start Guide](libcll/docs/QUICKSTART_ERROR_ASYMMETRY.md)**: Get up and running quickly  
- **[API Documentation](libcll/docs/)**: Complete function and class references
- **Examples**: Comprehensive usage examples and tutorials

## 🔍 Key Files

### Core Framework
- `libcll/metrics/error_asymmetry.py` - Core metric computation
- `libcll/metrics/visualize.py` - Visualization utilities
- `libcll/strategies/Strategy.py` - Automatic integration with training
- `libcll/datasets/` - Dataset loaders with noise injection
- `libcll/models/` - Neural network architectures

### Quality Diversity Integration
- `pyribs-master/ribs/archives/` - QD archives (Grid, CVT, etc.)
- `pyribs-master/ribs/emitters/` - Evolution strategies
- `pyribs-master/ribs/schedulers/` - QD algorithm coordination

### Examples and Tutorials
- `libcll/examples/error_asymmetry_example.py` - Basic usage examples
- `libcll/examples/qd_integration_example.py` - QD optimization demo
- `libcll/examples/qd_cma_algorithms.py` - CMA-ES/CMA-ME examples
- `libcll/examples/qd_real_training.py` - Real model training
- `libcll/examples/qd_simple_cpu.py` - CPU-only training

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **pyribs**: QD optimization framework by ICAROS Lab, USC
- **PyTorch Lightning**: Training infrastructure  
- **MNIST/CIFAR datasets**: Standard benchmarks for evaluation

## 📞 Support

- **Issues**: GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions for questions and ideas
- **Documentation**: Comprehensive guides in `libcll/docs/` folder
- **Examples**: Working code in `libcll/examples/` folder

---

## 🎯 Why CLLEA?

Traditional continual learning approaches focus on preventing catastrophic forgetting but often ignore **fairness across classes**. CLLEA addresses this by:

1. **Quantifying Bias**: Error asymmetry metrics reveal hidden model biases
2. **Promoting Diversity**: QD optimization maintains populations of diverse models  
3. **Enabling Choice**: Users can select models based on their fairness requirements
4. **Real Training**: No simulations - actual PyTorch model training and evaluation
5. **Comprehensive Analysis**: Rich visualizations and reports for understanding model behavior

**Perfect for**: Researchers in continual learning, fairness in ML, quality diversity, and multi-objective optimization who need to understand and control model behavior across different classes.
