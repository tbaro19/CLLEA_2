"""Integration example: Using Error Asymmetry BD with pyribs QD framework.

This script demonstrates how to integrate Error Asymmetry behavior descriptors
with the pyribs Quality Diversity optimization framework for continual learning.

The example shows:
1. Setting up a QD archive with Error Asymmetry BDs
2. Training multiple models with different configurations
3. Adding models to the archive based on accuracy and variance
4. Visualizing the QD map to identify generalist vs specialist models
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple

# Import pyribs components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "pyribs-master"))

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

# Import libcll components
from libcll.metrics import compute_error_asymmetry
from libcll.metrics.visualize import plot_qd_map_2d, plot_qd_heatmap


def create_qd_archive_for_cll(
    solution_dim: int,
    accuracy_range: Tuple[float, float] = (0.0, 1.0),
    variance_range: Tuple[float, float] = (0.0, 0.2),
    grid_dims: Tuple[int, int] = (50, 50),
) -> GridArchive:
    """Create a QD archive for continual learning with Error Asymmetry BDs.
    
    Parameters
    ----------
    solution_dim : int
        Dimension of the solution space (e.g., number of model parameters)
    accuracy_range : tuple
        Range for overall accuracy (measure 1)
    variance_range : tuple
        Range for accuracy variance (measure 2, BD)
    grid_dims : tuple
        Grid dimensions for the archive
        
    Returns
    -------
    archive : GridArchive
        Configured QD archive
    """
    archive = GridArchive(
        solution_dim=solution_dim,
        dims=grid_dims,
        ranges=[accuracy_range, variance_range],
        seed=42,
    )
    
    print(f"Created QD Archive:")
    print(f"  Grid: {grid_dims[0]}x{grid_dims[1]} = {grid_dims[0]*grid_dims[1]} cells")
    print(f"  Accuracy range: {accuracy_range}")
    print(f"  Variance range (BD): {variance_range}")
    print()
    
    return archive


def simulate_model_evaluation(
    model_params: np.ndarray,
    num_classes: int = 10,
    num_samples: int = 1000,
) -> Tuple[float, float, np.ndarray]:
    """Simulate model evaluation and compute Error Asymmetry metrics.
    
    In a real scenario, this would:
    1. Load a model with the given parameters
    2. Evaluate on a test set
    3. Compute predictions and metrics
    
    For this example, we simulate the behavior.
    
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
    overall_accuracy : float
        Overall test accuracy (objective)
    accuracy_variance : float
        Variance of per-class accuracies (BD)
    per_class_acc : np.ndarray
        Per-class accuracies
    """
    # Simulate predictions based on model parameters
    # In reality, you would evaluate your actual model here
    
    # Use model parameters to determine bias
    bias_factor = np.mean(model_params[:num_classes]) if len(model_params) >= num_classes else 0.5
    
    # Generate synthetic per-class accuracies
    np.random.seed(int(np.sum(np.abs(model_params)) * 1000) % 2**32)
    
    # Create biased accuracies
    per_class_acc = np.random.uniform(0.3, 0.95, num_classes)
    
    # Adjust based on bias factor
    if bias_factor > 0.6:
        # More specialist: make some classes very good, others poor
        per_class_acc[:num_classes//2] *= 1.2
        per_class_acc[num_classes//2:] *= 0.7
        per_class_acc = np.clip(per_class_acc, 0.0, 1.0)
    elif bias_factor < 0.4:
        # More generalist: make all classes similar
        mean_acc = per_class_acc.mean()
        per_class_acc = mean_acc + (per_class_acc - mean_acc) * 0.3
    
    # Compute overall accuracy and variance
    overall_accuracy = float(per_class_acc.mean())
    accuracy_variance = float(per_class_acc.var())
    
    return overall_accuracy, accuracy_variance, per_class_acc


def run_qd_optimization_for_cll(
    num_iterations: int = 100,
    num_classes: int = 10,
    solution_dim: int = 50,
    output_dir: str = "qd_cll_output",
):
    """Run QD optimization for continual learning with Error Asymmetry.
    
    Parameters
    ----------
    num_iterations : int
        Number of QD iterations
    num_classes : int
        Number of classes in the CLL problem
    solution_dim : int
        Dimension of the solution (model parameters)
    output_dir : str
        Output directory for results
    """
    print("=" * 70)
    print("QD Optimization for Continual Learning with Error Asymmetry")
    print("=" * 70)
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create QD archive
    archive = create_qd_archive_for_cll(
        solution_dim=solution_dim,
        accuracy_range=(0.0, 1.0),
        variance_range=(0.0, 0.2),
        grid_dims=(50, 50),
    )
    
    # Create emitter
    initial_solution = np.zeros(solution_dim)
    emitter = EvolutionStrategyEmitter(
        archive=archive,
        x0=initial_solution,
        sigma0=0.5,
        ranker="2imp",
        batch_size=36,
        seed=42,
    )
    
    # Create scheduler
    scheduler = Scheduler(archive=archive, emitters=[emitter])
    
    print("Starting QD optimization...")
    print()
    
    # Tracking
    all_accuracies = []
    all_variances = []
    all_objectives = []
    
    # Main QD loop
    for itr in range(num_iterations):
        # Ask for solutions
        solutions = scheduler.ask()
        
        # Evaluate solutions
        objectives = []
        measures = []
        
        for solution in solutions:
            # Simulate model evaluation
            accuracy, variance, per_class_acc = simulate_model_evaluation(
                solution, num_classes=num_classes
            )
            
            # Objective: maximize overall accuracy
            objectives.append(accuracy)
            
            # Measures: [accuracy, variance]
            measures.append([accuracy, variance])
            
            # Track for visualization
            all_accuracies.append(accuracy)
            all_variances.append(variance)
            all_objectives.append(accuracy)
        
        # Tell scheduler the results
        scheduler.tell(
            objective=np.array(objectives),
            measures=np.array(measures),
        )
        
        # Log progress
        if (itr + 1) % 10 == 0 or itr == 0:
            stats = archive.stats
            print(f"Iteration {itr + 1}/{num_iterations}")
            print(f"  Archive size: {stats.num_elites}")
            print(f"  Coverage: {stats.coverage * 100:.2f}%")
            print(f"  QD Score: {stats.qd_score:.4f}")
            print(f"  Best objective: {stats.obj_max:.4f}")
            print()
    
    print("QD optimization completed!")
    print()
    
    # Final statistics
    stats = archive.stats
    print("=" * 70)
    print("Final Archive Statistics")
    print("=" * 70)
    print(f"Total solutions found: {stats.num_elites}")
    print(f"Coverage: {stats.coverage * 100:.2f}%")
    print(f"QD Score: {stats.qd_score:.4f}")
    print(f"Best objective: {stats.obj_max:.4f}")
    print(f"Mean objective: {stats.obj_mean:.4f}")
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Get archive data
    df = archive.data(return_type="pandas")
    
    if len(df) > 0:
        # Extract measures
        archive_accuracies = df["measure_0"].values
        archive_variances = df["measure_1"].values
        archive_objectives = df["objective"].values
        
        # Plot QD map
        plot_qd_map_2d(
            accuracy=archive_accuracies,
            behavior_descriptor=archive_variances,
            values=archive_objectives,
            xlabel="Overall Accuracy",
            ylabel="Accuracy Variance (Error Asymmetry BD)",
            title=f"QD Map: Error Asymmetry (Archive Coverage: {stats.coverage*100:.1f}%)",
            save_path=str(output_path / "qd_map_archive.png"),
        )
        
        # Plot QD heatmap
        plot_qd_heatmap(
            accuracy=archive_accuracies,
            behavior_descriptor=archive_variances,
            values=archive_objectives,
            bins=(50, 50),
            xlabel="Overall Accuracy",
            ylabel="Accuracy Variance (Error Asymmetry BD)",
            title="QD Heatmap: Best Solutions in Each Cell",
            save_path=str(output_path / "qd_heatmap_archive.png"),
        )
        
        # Plot all evaluated solutions (including non-elites)
        plot_qd_map_2d(
            accuracy=np.array(all_accuracies),
            behavior_descriptor=np.array(all_variances),
            values=np.array(all_objectives),
            xlabel="Overall Accuracy",
            ylabel="Accuracy Variance (Error Asymmetry BD)",
            title="QD Map: All Evaluated Solutions",
            save_path=str(output_path / "qd_map_all.png"),
        )
        
        # Save archive to CSV
        df.to_csv(output_path / "archive.csv", index=False)
        
        print(f"✓ Visualizations saved to: {output_path}")
        print()
        
        # Analyze elite solutions
        print("=" * 70)
        print("Elite Solution Analysis")
        print("=" * 70)
        
        # Find generalists (low variance, high accuracy)
        generalists = df[(df["measure_1"] < 0.02) & (df["measure_0"] > 0.75)]
        print(f"Generalists (low variance, high accuracy): {len(generalists)}")
        if len(generalists) > 0:
            print(f"  Best generalist accuracy: {generalists['measure_0'].max():.4f}")
        
        # Find specialists (high variance, high accuracy)
        specialists = df[(df["measure_1"] > 0.05) & (df["measure_0"] > 0.75)]
        print(f"Specialists (high variance, high accuracy): {len(specialists)}")
        if len(specialists) > 0:
            print(f"  Best specialist accuracy: {specialists['measure_0'].max():.4f}")
        
        # Find balanced solutions
        balanced = df[(df["measure_1"] >= 0.02) & (df["measure_1"] <= 0.05)]
        print(f"Balanced solutions (moderate variance): {len(balanced)}")
        
        print()
    else:
        print("Warning: Archive is empty!")
    
    print("=" * 70)
    print("QD optimization completed successfully!")
    print("=" * 70)


def main():
    """Main function."""
    # Configure experiment
    config = {
        "num_iterations": 100,
        "num_classes": 10,
        "solution_dim": 50,
        "output_dir": "qd_cll_output",
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Run QD optimization
    run_qd_optimization_for_cll(**config)
    
    print("\nTo visualize results:")
    print(f"  - QD Map (Archive): qd_cll_output/qd_map_archive.png")
    print(f"  - QD Heatmap: qd_cll_output/qd_heatmap_archive.png")
    print(f"  - All Solutions: qd_cll_output/qd_map_all.png")
    print(f"  - Archive Data: qd_cll_output/archive.csv")
    print()


if __name__ == "__main__":
    main()
