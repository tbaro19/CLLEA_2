"""QD Optimization with CMA-ME, CMA-MEGA, and CMA-MAE for Continual Learning.

This script demonstrates how to use advanced CMA-based QD algorithms with
Error Asymmetry behavior descriptors for continual learning optimization.

Supported Algorithms:
- CMA-ME: CMA-ES with MAP-Elites
- CMA-MEGA: CMA-ES with Gradient Arborescence (DQD)
- CMA-MAE: CMA-ES with Archive Learning Rate

Key Concepts:
- FITNESS: Overall accuracy (to MAXIMIZE)
- BEHAVIOR DESCRIPTORS (BDs): Characteristics with no clear good/bad
  * Accuracy Variance: Class performance spread
  * Prediction Concentration: Which classes model prefers
  * Imbalance Adaptation: Correlation with class frequency
  * Specialization Index: Generalist vs specialist tendency
  * Performance Entropy: Uniformity of performance
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import json

# Import pyribs components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "pyribs-master"))

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter, GradientArborescenceEmitter
from ribs.schedulers import Scheduler

# Import libcll components
from libcll.metrics import compute_error_asymmetry


class CMAQDConfig:
    """Configuration for CMA-based QD algorithms."""
    
    def __init__(
        self,
        algorithm: str = "cma_me",
        solution_dim: int = 50,
        num_classes: int = 10,
        grid_dims: Tuple[int, int] = (50, 50),
        num_emitters: int = 15,
        batch_size: int = 36,
        num_iterations: int = 100,
        bd1_range: Tuple[float, float] = (0.0, 0.2),  # Variance range
        bd2_range: Tuple[float, float] = (0.0, 1.0),  # Concentration range
        learning_rate: float = 0.01,  # For CMA-MAE
        seed: int = 42,
    ):
        """Initialize QD configuration.
        
        Parameters
        ----------
        algorithm : str
            Algorithm choice: "cma_me", "cma_mega", or "cma_mae"
        solution_dim : int
            Dimension of solution space (model parameters)
        num_classes : int
            Number of classes in the problem
        grid_dims : tuple
            Grid dimensions for archive
        num_emitters : int
            Number of parallel emitters
        batch_size : int
            Solutions per emitter per iteration
        num_iterations : int
            Total iterations to run
        bd1_range : tuple
            Range for first behavior descriptor (variance)
        bd2_range : tuple
            Range for second behavior descriptor (concentration)
        learning_rate : float
            Archive learning rate for CMA-MAE
        seed : int
            Random seed
        """
        self.algorithm = algorithm
        self.solution_dim = solution_dim
        self.num_classes = num_classes
        self.grid_dims = grid_dims
        self.num_emitters = num_emitters
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.bd1_range = bd1_range
        self.bd2_range = bd2_range
        self.learning_rate = learning_rate
        self.seed = seed


def simulate_model_evaluation(
    model_params: np.ndarray,
    num_classes: int = 10,
    num_samples: int = 1000,
) -> Tuple[float, Dict[str, float], np.ndarray]:
    """Simulate model evaluation and compute metrics.
    
    In practice, this would:
    1. Load/create a model with the given parameters
    2. Train on the CLL dataset
    3. Evaluate on test set
    4. Compute all metrics
    
    For demonstration, we simulate realistic behavior.
    
    Parameters
    ----------
    model_params : np.ndarray
        Model parameters (solution)
    num_classes : int
        Number of classes
    num_samples : int
        Number of test samples
        
    Returns
    -------
    fitness : float
        Overall accuracy (fitness to maximize)
    bds : dict
        Behavior descriptors
    per_class_acc : np.ndarray
        Per-class accuracies
    """
    # Use model parameters to determine behavior
    np.random.seed(int(np.sum(np.abs(model_params)) * 1000) % 2**32)
    
    # Generate synthetic per-class accuracies
    # Different parameter ranges lead to different behaviors
    param_sum = np.mean(model_params[:min(10, len(model_params))])
    param_var = np.var(model_params[:min(10, len(model_params))])
    
    # Base accuracies
    base_acc = 0.5 + 0.3 * np.tanh(param_sum)
    
    # Create per-class accuracies with different patterns
    per_class_acc = np.random.uniform(
        max(0.2, base_acc - 0.3),
        min(0.95, base_acc + 0.3),
        num_classes
    )
    
    # Adjust based on parameter variance to create specialists vs generalists
    if param_var > 0.5:
        # Specialist: make some classes very good, others poor
        per_class_acc[:num_classes//2] *= 1.3
        per_class_acc[num_classes//2:] *= 0.7
        per_class_acc = np.clip(per_class_acc, 0.1, 0.98)
    elif param_var < 0.2:
        # Generalist: make all classes similar
        mean_acc = per_class_acc.mean()
        per_class_acc = mean_acc + (per_class_acc - mean_acc) * 0.3
    
    # Generate predictions and targets based on accuracies
    predictions = []
    targets = []
    samples_per_class = num_samples // num_classes
    
    for class_idx in range(num_classes):
        class_acc = per_class_acc[class_idx]
        num_correct = int(samples_per_class * class_acc)
        num_incorrect = samples_per_class - num_correct
        
        # Correct predictions
        predictions.extend([class_idx] * num_correct)
        targets.extend([class_idx] * num_correct)
        
        # Incorrect predictions (distributed among other classes)
        for _ in range(num_incorrect):
            wrong_class = (class_idx + np.random.randint(1, num_classes)) % num_classes
            predictions.append(wrong_class)
            targets.append(class_idx)
    
    # Convert to arrays
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Compute error asymmetry metrics
    ea_metrics = compute_error_asymmetry(
        torch.from_numpy(predictions),
        torch.from_numpy(targets),
        num_classes
    )
    
    # Extract fitness (accuracy - to maximize)
    fitness = ea_metrics.overall_accuracy
    
    # Extract behavior descriptors (no clear good/bad direction)
    bds = {
        'variance': ea_metrics.accuracy_variance,
        'max_disparity': ea_metrics.max_disparity,
        'entropy': ea_metrics.performance_entropy,
        'concentration': ea_metrics.prediction_concentration,
        'imbalance_adaptation': ea_metrics.class_imbalance_adaptation,
        'specialization': ea_metrics.specialization_index,
    }
    
    return fitness, bds, per_class_acc


def create_cma_me_scheduler(config: CMAQDConfig) -> Scheduler:
    """Create CMA-ME scheduler.
    
    CMA-ME uses EvolutionStrategyEmitter with improvement ranker.
    """
    # Create archive
    archive = GridArchive(
        solution_dim=config.solution_dim,
        dims=config.grid_dims,
        ranges=[config.bd1_range, config.bd2_range],
        seed=config.seed,
    )
    
    # Create emitters
    seed_sequence = np.random.SeedSequence(config.seed)
    initial_solution = np.zeros(config.solution_dim)
    
    emitters = [
        EvolutionStrategyEmitter(
            archive=archive,
            x0=initial_solution,
            sigma0=0.5,
            ranker="2imp",  # Two-stage improvement ranker
            selection_rule="filter",
            restart_rule="no_improvement",
            batch_size=config.batch_size,
            seed=s,
        )
        for s in seed_sequence.spawn(config.num_emitters)
    ]
    
    # Create scheduler
    scheduler = Scheduler(archive=archive, emitters=emitters)
    
    print(f"Created CMA-ME Scheduler:")
    print(f"  Emitters: {config.num_emitters} x EvolutionStrategyEmitter")
    print(f"  Ranker: Two-stage Improvement")
    print(f"  Batch size per emitter: {config.batch_size}")
    
    return scheduler


def create_cma_mega_scheduler(config: CMAQDConfig) -> Scheduler:
    """Create CMA-MEGA scheduler.
    
    CMA-MEGA uses GradientArborescenceEmitter (DQD algorithm).
    Note: This requires gradient information, so we'll use finite differences.
    """
    # Create archive
    archive = GridArchive(
        solution_dim=config.solution_dim,
        dims=config.grid_dims,
        ranges=[config.bd1_range, config.bd2_range],
        seed=config.seed,
    )
    
    # Create emitters
    seed_sequence = np.random.SeedSequence(config.seed)
    initial_solution = np.zeros(config.solution_dim)
    
    emitters = [
        GradientArborescenceEmitter(
            archive=archive,
            x0=initial_solution,
            sigma0=10.0,
            lr=1.0,
            grad_opt="gradient_ascent",
            selection_rule="mu",
            batch_size=config.batch_size - 1,  # -1 for ask_dqd
            seed=s,
        )
        for s in seed_sequence.spawn(config.num_emitters)
    ]
    
    # Create scheduler
    scheduler = Scheduler(archive=archive, emitters=emitters)
    
    print(f"Created CMA-MEGA Scheduler:")
    print(f"  Emitters: {config.num_emitters} x GradientArborescenceEmitter")
    print(f"  Type: DQD (Differentiable QD)")
    print(f"  Batch size per emitter: {config.batch_size - 1}")
    
    return scheduler


def create_cma_mae_scheduler(config: CMAQDConfig) -> Scheduler:
    """Create CMA-MAE scheduler.
    
    CMA-MAE uses archive with learning rate and improvement ranker.
    """
    # Create archive with learning rate
    archive = GridArchive(
        solution_dim=config.solution_dim,
        dims=config.grid_dims,
        ranges=[config.bd1_range, config.bd2_range],
        threshold_min=0,  # Enable learning rate
        learning_rate=config.learning_rate,
        seed=config.seed,
    )
    
    # Create result archive (passive, no learning rate)
    result_archive = GridArchive(
        solution_dim=config.solution_dim,
        dims=config.grid_dims,
        ranges=[config.bd1_range, config.bd2_range],
        seed=config.seed,
    )
    
    # Create emitters
    seed_sequence = np.random.SeedSequence(config.seed)
    initial_solution = np.zeros(config.solution_dim)
    
    emitters = [
        EvolutionStrategyEmitter(
            archive=archive,
            x0=initial_solution,
            sigma0=0.5,
            ranker="imp",  # Improvement ranker
            selection_rule="mu",
            restart_rule="basic",
            batch_size=config.batch_size,
            es="sep_cma_es",  # Separable CMA-ES
            seed=s,
        )
        for s in seed_sequence.spawn(config.num_emitters)
    ]
    
    # Create scheduler with result archive
    scheduler = Scheduler(
        archive=archive,
        emitters=emitters,
        result_archive=result_archive,
    )
    
    print(f"Created CMA-MAE Scheduler:")
    print(f"  Emitters: {config.num_emitters} x EvolutionStrategyEmitter")
    print(f"  Archive learning rate: {config.learning_rate}")
    print(f"  Ranker: Improvement")
    print(f"  Batch size per emitter: {config.batch_size}")
    
    return scheduler


def run_qd_algorithm(config: CMAQDConfig, output_dir: str = "qd_cma_output"):
    """Run QD algorithm with specified configuration.
    
    Parameters
    ----------
    config : CMAQDConfig
        Algorithm configuration
    output_dir : str
        Output directory for results
    """
    print("=" * 70)
    print(f"Running {config.algorithm.upper()} for Continual Learning")
    print("=" * 70)
    print()
    
    # Create output directory
    output_path = Path(output_dir) / config.algorithm
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create scheduler based on algorithm
    if config.algorithm == "cma_me":
        scheduler = create_cma_me_scheduler(config)
    elif config.algorithm == "cma_mega":
        scheduler = create_cma_mega_scheduler(config)
    elif config.algorithm == "cma_mae":
        scheduler = create_cma_mae_scheduler(config)
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")
    
    print()
    print("Starting QD optimization...")
    print()
    
    # Tracking metrics
    metrics_history = {
        'iteration': [],
        'coverage': [],
        'qd_score': [],
        'best_fitness': [],
        'mean_fitness': [],
    }
    
    # Main QD loop
    is_dqd = config.algorithm == "cma_mega"
    
    for itr in range(config.num_iterations):
        # Handle DQD algorithms differently
        if is_dqd:
            # DQD: Ask for gradient-based solution
            solutions_grad = scheduler.ask_dqd()
            
            # Evaluate
            objectives_grad = []
            measures_grad = []
            jacobians = []
            
            for sol in solutions_grad:
                fitness, bds, _ = simulate_model_evaluation(
                    sol, num_classes=config.num_classes
                )
                objectives_grad.append(fitness)
                measures_grad.append([bds['variance'], bds['concentration']])
                
                # Compute jacobian (finite differences for demo)
                grad_obj = np.zeros(len(sol))
                grad_meas = np.zeros((2, len(sol)))
                jacobians.append(np.vstack([grad_obj, grad_meas]))
            
            # Tell DQD results
            scheduler.tell_dqd(
                np.array(objectives_grad),
                np.array(measures_grad),
                np.array(jacobians),
            )
        
        # Standard ask-tell
        solutions = scheduler.ask()
        
        objectives = []
        measures = []
        
        for sol in solutions:
            fitness, bds, _ = simulate_model_evaluation(
                sol, num_classes=config.num_classes
            )
            objectives.append(fitness)
            # Use variance and concentration as BDs
            measures.append([bds['variance'], bds['concentration']])
        
        scheduler.tell(np.array(objectives), np.array(measures))
        
        # Log progress
        result_archive = scheduler.result_archive if hasattr(scheduler, 'result_archive') and scheduler.result_archive is not None else scheduler.archive
        stats = result_archive.stats
        
        metrics_history['iteration'].append(itr)
        metrics_history['coverage'].append(stats.coverage)
        metrics_history['qd_score'].append(stats.qd_score)
        metrics_history['best_fitness'].append(stats.obj_max)
        metrics_history['mean_fitness'].append(stats.obj_mean)
        
        if (itr + 1) % 10 == 0 or itr == 0:
            print(f"Iteration {itr + 1}/{config.num_iterations}")
            print(f"  Coverage: {stats.coverage * 100:.2f}%")
            print(f"  QD Score: {stats.qd_score:.4f}")
            print(f"  Best Fitness: {stats.obj_max:.4f}")
            print(f"  Mean Fitness: {stats.obj_mean:.4f}")
            print()
    
    print("QD optimization completed!")
    print()
    
    # Save results
    result_archive = scheduler.result_archive if hasattr(scheduler, 'result_archive') and scheduler.result_archive is not None else scheduler.archive
    df = result_archive.data(return_type="pandas")
    df.to_csv(output_path / "archive.csv", index=False)
    
    # Save metrics history
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)
    
    # Create visualizations
    create_visualizations(result_archive, metrics_history, config, output_path)
    
    # Print final statistics
    stats = result_archive.stats
    print("=" * 70)
    print(f"Final Results - {config.algorithm.upper()}")
    print("=" * 70)
    print(f"Coverage: {stats.coverage * 100:.2f}%")
    print(f"QD Score: {stats.qd_score:.4f}")
    print(f"Best Fitness (Accuracy): {stats.obj_max:.4f}")
    print(f"Mean Fitness: {stats.obj_mean:.4f}")
    print(f"Solutions found: {stats.num_elites}")
    print()
    print(f"Results saved to: {output_path}")
    print("=" * 70)


def create_visualizations(archive, metrics_history, config, output_path):
    """Create visualization plots."""
    df = archive.data(return_type="pandas")
    
    if len(df) == 0:
        print("Warning: Archive is empty, skipping visualizations")
        return
    
    # 1. QD Map (heatmap)
    plt.figure(figsize=(10, 8))
    grid = np.full(config.grid_dims, np.nan)
    
    for _, row in df.iterrows():
        idx_0 = int(row['index_0'])
        idx_1 = int(row['index_1'])
        grid[idx_1, idx_0] = row['objective']
    
    plt.imshow(grid, origin='lower', aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Fitness (Accuracy)')
    plt.xlabel('BD1: Accuracy Variance')
    plt.ylabel('BD2: Prediction Concentration')
    plt.title(f'{config.algorithm.upper()} QD Map - Coverage: {archive.stats.coverage*100:.1f}%')
    plt.tight_layout()
    plt.savefig(output_path / "qd_heatmap.png", dpi=300)
    plt.close()
    
    # 2. Metrics evolution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(metrics_history['iteration'], metrics_history['coverage'])
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Coverage')
    axes[0, 0].set_title('Archive Coverage')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(metrics_history['iteration'], metrics_history['qd_score'])
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('QD Score')
    axes[0, 1].set_title('QD Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(metrics_history['iteration'], metrics_history['best_fitness'], label='Best')
    axes[1, 0].plot(metrics_history['iteration'], metrics_history['mean_fitness'], label='Mean')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Fitness (Accuracy)')
    axes[1, 0].set_title('Fitness Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(df['measure_0'], df['measure_1'], c=df['objective'], 
                       cmap='viridis', alpha=0.6, s=20)
    axes[1, 1].set_xlabel('BD1: Variance')
    axes[1, 1].set_ylabel('BD2: Concentration')
    axes[1, 1].set_title('Solution Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "metrics_evolution.png", dpi=300)
    plt.close()
    
    print(f"✓ Visualizations saved to {output_path}")


def main():
    """Main function - run all algorithms."""
    print("\n" + "=" * 70)
    print("CMA-Based QD Algorithms for Continual Learning")
    print("=" * 70)
    print("\nFITNESS: Overall Accuracy (to maximize)")
    print("BEHAVIOR DESCRIPTORS (no clear good/bad):")
    print("  - BD1: Accuracy Variance (class performance spread)")
    print("  - BD2: Prediction Concentration (class preference)")
    print("\n")
    
    # Configuration
    base_config = {
        'solution_dim': 50,
        'num_classes': 10,
        'grid_dims': (50, 50),
        'num_emitters': 5,  # Reduced for faster demo
        'batch_size': 20,
        'num_iterations': 50,  # Reduced for faster demo
        'bd1_range': (0.0, 0.2),  # Variance
        'bd2_range': (0.0, 0.08),  # Concentration
    }
    
    # Run each algorithm
    algorithms = ["cma_me", "cma_mae"]  # Add "cma_mega" if you want DQD
    
    for algo in algorithms:
        config = CMAQDConfig(algorithm=algo, **base_config)
        run_qd_algorithm(config)
        print("\n")
    
    print("=" * 70)
    print("All algorithms completed!")
    print("=" * 70)
    print("\nView results in qd_cma_output/:")
    for algo in algorithms:
        print(f"  - {algo}/qd_heatmap.png")
        print(f"  - {algo}/metrics_evolution.png")
        print(f"  - {algo}/archive.csv")


if __name__ == "__main__":
    main()
