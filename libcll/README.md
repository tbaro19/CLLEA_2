# libcll: Complementary Label Learning Benchmark

[![Documentation Status](https://readthedocs.org/projects/libcll/badge/?version=latest)](https://libcll.readthedocs.io/en/latest/?badge=latest) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img src="docs/libcll-cover.png" alt="libcll" style="zoom:25%;" />

`libcll` is a Python library designed to simplify complementary-label learning (CLL) for researchers tackling real-world challenges. The package implements a wide range of popular CLL strategies, including **CPE**, the state-of-the-art algorithm as of 2023. Additionally, it includes unique datasets like **CLImage** and **ACLImage**, which feature complementary labels collected from human annotators and VLM annotators. To foster extensibility, `libcll` provides a unified interface for integrating additional strategies, datasets, and models, making it a versatile tool for advancing CLL research. For more details, refer to the associated technical report on [arXiv](https://arxiv.org/abs/2411.12276).

# Installation

- Python version >= 3.8, <= 3.12
- Pytorch version >= 1.11, <= 2.0
- Pytorch Lightning version >= 2.0
- To install `libcll` and develop locally:

```
git clone git@github.com:ntucllab/libcll.git
cd libcll
pip install -e .
```

# Running

## Supported Strategies

| Strategies                                                 | Type             | Description                                                  |
| ---------------------------------------------------------- | ---------------- | ------------------------------------------------------------ |
| [PC](https://arxiv.org/pdf/1705.07541)                     | None             | Pairwise-Comparison Loss                                     |
| [SCL](https://arxiv.org/pdf/2007.02235.pdf)                | NL, EXP          | Surrogate Complementary Loss with the negative log loss (NL) or with the exponential loss (EXP) |
| [URE](https://arxiv.org/pdf/1810.04327.pdf)                | NN, GA, TNN, TGA | Unbiased Risk Estimator whether with gradient ascent (GA) or empirical transition matrix (T) |
| [FWD](https://arxiv.org/pdf/1711.09535.pdf)                | None             | Forward Correction                                           |
| [DM](http://proceedings.mlr.press/v139/gao21d/gao21d.pdf)  | None             | Discriminative Models with Weighted Loss                     |
| [CPE](https://arxiv.org/pdf/2209.09500.pdf)                | I, F, T          | Complementary Probability Estimates with different transition matrices (I, F, T) |
| [MCL](https://arxiv.org/pdf/1912.12927.pdf)                | MAE, EXP, LOG    | Multiple Complementary Label learning with different errors (MAE, EXP, LOG) |
| [OP](https://proceedings.mlr.press/v206/liu23g/liu23g.pdf) | None             | Order-Preserving Loss                                        |
| [SCARCE](https://arxiv.org/pdf/2311.15502)                 | None             | Selected-Completely-At-Random Complementary-label learning   |

## Supported Datasets

| Dataset     | Number of Classes | Input Size  | Description                                                  |
| ----------- | --------------- | ----------- | ------------------------------------------------------------ |
| MNIST       | 10              | 28 x 28     | Grayscale images of handwritten digits (0 to 9).             |
| FMNIST      | 10              | 28 x 28     | Grayscale images of fashion items.                           |
| KMNIST      | 10              | 28 x 28     | Grayscale images of cursive Japanese (“Kuzushiji”) characters. |
| [Yeast](https://www.openml.org/search?type=data&status=active&id=181) | 10              | 8           | Features of different localization sites of protein.         |
| [Texture](https://www.openml.org/search?type=data&status=active&id=40499) | 11              | 40          | Features of different textures.                              |
| [Dermatology](https://www.openml.org/search?type=data&status=active&id=35) | 6               | 130         | Clinical Attributes of different diseases.                              |
| [Control](https://www.openml.org/search?type=data&status=active&id=377) | 6               | 60          | Features of synthetically generated control charts.          |
| CIFAR10 | 10 | 3 x 32 x 32 | Colored images of different objects. |
| CIFAR20     | 20              | 3 x 32 x 32 | Colored images of different objects. |
| Micro ImageNet10   | 10                | 3 x 64 x 64 | Contains images of 10 classes designed for computer vision research. |
| Micro ImageNet20 | 20 | 3 x 64 x 64 | Contains images of 20 classes designed for computer vision research. |
| CLCIFAR10   | 10              | 3 x 32 x 32 | Colored images of distinct objects paired with complementary labels annotated by humans. |
| CLCIFAR20   | 20              | 3 x 32 x 32 | Colored images of distinct objects paired with complementary labels annotated by humans. |
| CLMicro ImageNet10 | 10 | 3 x 64 x 64 | Contains images of 10 classes designed for computer vision research paired with complementary labels annotated by humans. |
| CLMicro ImageNet20 | 20 | 3 x 64 x 64 | Contains images of 20 classes designed for computer vision research paired with complementary labels annotated by humans. |
| ACLCIFAR10   | 10              | 3 x 32 x 32 | Colored images of distinct objects paired with complementary labels annotated by Visual-Language Models. |
| ACLCIFAR20   | 20              | 3 x 32 x 32 | Colored images of distinct objects paired with complementary labels annotated by Visual-Language Models. |
| ACLMicro ImageNet10 | 10 | 3 x 64 x 64 | Contains images of 10 classes designed for computer vision research paired with complementary labels annotated by Visual-Language Models. |
| ACLMicro ImageNet20 | 20 | 3 x 64 x 64 | Contains images of 20 classes designed for computer vision research paired with complementary labels annotated by Visual-Language Models. |

## Quick Start: Complementary Label Learning on MNIST

To reproduce training results with the SCL-NL method on MNIST for each distribution:

### Uniform Distribution

```shell
python scripts/train.py \
  --do_train \
  --do_predict \
  --strategy SCL \
  --type NL \
  --model MLP \
  --dataset MNIST \
  --lr 1e-4 \
  --batch_size 256 \
  --valid_type Accuracy \
```

### Biased Distribution (Weak Deviation)

```shell
python scripts/train.py \
  --do_train \
  --do_predict \
  --strategy SCL \
  --type NL \
  --model MLP \
  --dataset MNIST \
  --lr 1e-4 \
  --batch_size 256 \
  --valid_type Accuracy \
  --transition_matrix weak
```
### Biased Distribution (Strong Deviation)

```shell
python scripts/train.py \
  --do_train \
  --do_predict \
  --strategy SCL \
  --type NL \
  --model MLP \
  --dataset MNIST \
  --lr 1e-4 \
  --batch_size 256 \
  --valid_type Accuracy \
  --transition_matrix strong
```

### Noisy Distribution

```shell
python scripts/train.py \
  --do_train \
  --do_predict \
  --strategy SCL \
  --type NL \
  --model MLP \
  --dataset MNIST \
  --lr 1e-4 \
  --batch_size 256 \
  --valid_type Accuracy \
  --transition_matrix noisy
  --noise 0.1
```
### Multiple Complementary Label Learning

```shell
python scripts/train.py \
  --do_train \
  --do_predict \
  --strategy SCL \
  --type NL \
  --model MLP \
  --dataset MNIST \
  --lr 1e-4 \
  --batch_size 256 \
  --valid_type Accuracy \
  --num_cl 3
```

## Run all the settings in the survey paper

The following scripts reproduce the results for one strategy presented in the survey paper. They include a grid search over learning rates from {1e-3, 5e-4, 1e-4, 5e-5, 1e-5}, followed by training with the best learning rate using four different random seeds.

```shell
./scripts/uniform.sh <strategy> <type>
./scripts/biased.sh <strategy> <type>
./scripts/noisy.sh <strategy> <type>
./scripts/multi.sh <strategy> <type>
./scripts/multi_hard.sh <strategy> <type>
```

For example:

```shell
./scripts/uniform.sh SCL NL
./scripts/biased.sh SCL NL
./scripts/noisy.sh SCL NL
./scripts/multi.sh SCL NL
./scripts/multi_hard.sh SCL NL
```

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

# Citing

If you find this package useful, please cite both the original works associated with each strategy and the following:

```
@techreport{libcll2024,
  author = {Nai-Xuan Ye and Tan-Ha Mai and Hsiu-Hsuan Wang and Wei-I Lin and Hsuan-Tien Lin},
  title = {libcll: an Extendable Python Toolkit for Complementary-Label Learning},
  institution = {National Taiwan University},
  url = {https://github.com/ntucllab/libcll},
  note = {available as arXiv preprint \url{https://arxiv.org/abs/2411.12276}},
  month = nov,
  year = 2024
}
```

# Acknowledgment

We would like to express our gratitude to the following repositories for sharing their code, which greatly facilitated the development of `libcll`:
* [URE and FWD implementation](https://github.com/takashiishida/comp)
* [DM official implementation](http://palm.seu.edu.cn/zhangml/Resources.htm#icml21b)
* [OP official implementation](https://github.com/yzcao-nkg/OPCLL)
* [SCARCE official implementation](https://github.com/wwangwitsel/SCARCE/tree/main)
* [CLImage Dataset implementation](https://github.com/ntucllab/CLImage_Dataset)
* [ACLImage Dataset implementation](https://github.com/yahcreepers/PAKDD_ACLImage_Dataset)
* [Code structure](https://github.com/ntucllab/imbalanced-DL)
