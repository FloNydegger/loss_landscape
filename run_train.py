"""
Main Training Script - Experiment Conductor

This script orchestrates the entire training process:
1. Loads data from generated dataset
2. Converts grid data to coordinate format
3. Initializes MLP models
4. Trains using both PINN and Data-Driven approaches
5. Saves results and visualizations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse

from data_loader import PoissonDataset, load_metadata
from models import MLP
from train import train_data_driven, train_pinn, evaluate_model


def grid_to_coords(N, domain=[0, 1]):
    """
    Convert grid indices to (x, y) coordinates.
    
    Parameters:
    N : int
        Grid size (N x N)
    domain : list
        Domain bounds [x_min, x_max]
        
    Returns:
    coords : np.ndarray
        Coordinate array of shape (N*N, 2)
    """
    x = np.linspace(domain[0], domain[1], N)
    y = np.linspace(domain[0], domain[1], N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    coords = np.stack([X.flatten(), Y.flatten()], axis=1)
    return coords


def prepare_data(sample, N=64, device='cpu'):
    """
    Prepare data from a sample for training.
    
    Parameters:
    sample : dict
        Sample dictionary from PoissonDataset
    N : int
        Grid size
    device : str
        Device to place tensors on
        
    Returns:
    coords : torch.Tensor
        Coordinates (N*N, 2)
    f_values : torch.Tensor
        Source term values (N*N, 1)
    u_exact : torch.Tensor
        Exact solution values (N*N, 1)
    """
    # Generate coordinates
    coords = grid_to_coords(N)
    coords = torch.tensor(coords, dtype=torch.float32, device=device)
    
    # Flatten field values
    f_values = sample['f'].flatten().unsqueeze(-1).to(device)
    u_exact = sample['u'].flatten().unsqueeze(-1).to(device)
    
    return coords, f_values, u_exact


def plot_training_history(history_data, history_pinn, K_value, save_path):
    """
    Plot training loss curves for both methods.
    
    Parameters:
    history_data : dict
        Training history from data-driven approach
    history_pinn : dict
        Training history from PINN approach
    K_value : int
        K value for the experiment
    save_path : Path
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training loss
    axes[0].semilogy(history_data['epoch'], history_data['loss'], 
                     label='Data-Driven', linewidth=2, alpha=0.8)
    axes[0].semilogy(history_pinn['epoch'], history_pinn['loss'], 
                     label='PINN', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title(f'Training Loss (K={K_value})', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: L2 relative error
    axes[1].semilogy(history_data['epoch'], history_data['l2_error'], 
                     label='Data-Driven', linewidth=2, alpha=0.8)
    axes[1].semilogy(history_pinn['epoch'], history_pinn['l2_error'], 
                     label='PINN', linewidth=2, alpha=0.8)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('L2 Relative Error', fontsize=12)
    axes[1].set_title(f'L2 Relative Error (K={K_value})', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved training history plot to {save_path}")


def plot_predictions(coords, u_exact, u_pred_data, u_pred_pinn, N, K_value, save_path):
    """
    Plot ground truth and predictions from both methods.
    
    Parameters:
    coords : torch.Tensor
        Coordinates
    u_exact : torch.Tensor
        Exact solution
    u_pred_data : np.ndarray
        Predictions from data-driven method
    u_pred_pinn : np.ndarray
        Predictions from PINN method
    N : int
        Grid size
    K_value : int
        K value for the experiment
    save_path : Path
        Path to save the figure
    """
    # Reshape to 2D grids
    u_exact_grid = u_exact.cpu().numpy().reshape(N, N)
    u_pred_data_grid = u_pred_data.reshape(N, N)
    u_pred_pinn_grid = u_pred_pinn.reshape(N, N)
    
    # Compute errors
    error_data = np.abs(u_exact_grid - u_pred_data_grid)
    error_pinn = np.abs(u_exact_grid - u_pred_pinn_grid)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Common colormap settings
    vmin = min(u_exact_grid.min(), u_pred_data_grid.min(), u_pred_pinn_grid.min())
    vmax = max(u_exact_grid.max(), u_pred_data_grid.max(), u_pred_pinn_grid.max())
    
    # Row 1: Solutions
    im0 = axes[0, 0].imshow(u_exact_grid, cmap='RdBu_r', origin='lower', 
                            extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Ground Truth u', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(u_pred_data_grid, cmap='RdBu_r', origin='lower',
                            extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Data-Driven Prediction', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(u_pred_pinn_grid, cmap='RdBu_r', origin='lower',
                            extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
    axes[0, 2].set_title('PINN Prediction', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Row 2: Errors
    axes[1, 0].axis('off')  # Empty plot for alignment
    
    error_max = max(error_data.max(), error_pinn.max())
    
    im3 = axes[1, 1].imshow(error_data, cmap='hot', origin='lower',
                            extent=[0, 1, 0, 1], vmin=0, vmax=error_max)
    axes[1, 1].set_title('Data-Driven Error', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 1])
    
    im4 = axes[1, 2].imshow(error_pinn, cmap='hot', origin='lower',
                            extent=[0, 1, 0, 1], vmin=0, vmax=error_max)
    axes[1, 2].set_title('PINN Error', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 2])
    
    plt.suptitle(f'Predictions and Errors (K={K_value})', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved predictions plot to {save_path}")


def run_experiment(data_dir, K_value, sample_idx=0, 
                  hidden_dim=64, num_hidden_layers=4,
                  num_epochs=5000, lr_adam=1e-3, lr_lbfgs=1.0, switch_epoch=3000,
                  lambda_bc=1.0, device='cpu', save_dir='results'):
    """
    Run a complete experiment for a given K value.
    
    Parameters:
    data_dir : str
        Directory containing the dataset
    K_value : int
        K value to use (1, 4, 8, or 16)
    sample_idx : int
        Which sample to use from the dataset
    hidden_dim : int
        Hidden layer dimension
    num_hidden_layers : int
        Number of hidden layers
    num_epochs : int
        Number of training epochs
    lr_adam : float
        Learning rate for Adam
    lr_lbfgs : float
        Learning rate for L-BFGS
    switch_epoch : int
        Epoch to switch optimizers
    lambda_bc : float
        Boundary condition weight for PINN
    device : str
        Device to use ('cpu' or 'cuda')
    save_dir : str
        Directory to save results
        
    Returns:
    results : dict
        Dictionary containing all results
    """
    print("\n" + "=" * 80)
    print(f"Running Experiment: K={K_value}, Sample={sample_idx}")
    print("=" * 80)
    
    # Create save directory
    save_path = Path(save_dir) / f"K_{K_value}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata = load_metadata(data_dir)
    N = metadata['N']
    
    # Load dataset
    print(f"\n1. Loading data...")
    dataset = PoissonDataset(data_dir, K_value)
    sample = dataset[sample_idx]
    
    # Prepare data
    print(f"2. Preparing data...")
    coords, f_values, u_exact = prepare_data(sample, N=N, device=device)
    print(f"   Coordinates shape: {coords.shape}")
    print(f"   Source term range: [{f_values.min():.4f}, {f_values.max():.4f}]")
    print(f"   Solution range: [{u_exact.min():.6f}, {u_exact.max():.6f}]")
    
    # Train Data-Driven model
    print(f"\n3. Training Data-Driven model...")
    model_data = MLP(input_dim=2, hidden_dim=hidden_dim, 
                     output_dim=1, num_hidden_layers=num_hidden_layers).to(device)
    print(f"   Model parameters: {model_data.count_parameters()}")
    
    history_data = train_data_driven(
        model_data, coords, f_values, u_exact,
        num_epochs=num_epochs, lr_adam=lr_adam, lr_lbfgs=lr_lbfgs,
        switch_epoch=switch_epoch, device=device, verbose=True
    )
    
    # Evaluate Data-Driven model
    metrics_data = evaluate_model(model_data, coords, u_exact, device=device)
    print(f"   Final L2 relative error: {metrics_data['l2_relative_error']:.6e}")
    
    # Train PINN model
    print(f"\n4. Training PINN model...")
    model_pinn = MLP(input_dim=2, hidden_dim=hidden_dim,
                     output_dim=1, num_hidden_layers=num_hidden_layers).to(device)
    print(f"   Model parameters: {model_pinn.count_parameters()}")
    
    history_pinn = train_pinn(
        model_pinn, coords, f_values, u_exact,
        num_epochs=num_epochs, lr_adam=lr_adam, lr_lbfgs=lr_lbfgs,
        switch_epoch=switch_epoch, lambda_bc=lambda_bc, device=device, verbose=True
    )
    
    # Evaluate PINN model
    metrics_pinn = evaluate_model(model_pinn, coords, u_exact, device=device)
    print(f"   Final L2 relative error: {metrics_pinn['l2_relative_error']:.6e}")
    
    # Save plots
    print(f"\n5. Generating visualizations...")
    plot_training_history(history_data, history_pinn, K_value, 
                          save_path / f"training_history_sample_{sample_idx}.png")
    plot_predictions(coords, u_exact, 
                    metrics_data['u_pred'], metrics_pinn['u_pred'],
                    N, K_value, save_path / f"predictions_sample_{sample_idx}.png")
    
    # Save results
    results = {
        'K_value': K_value,
        'sample_idx': sample_idx,
        'N': N,
        'model_config': {
            'hidden_dim': hidden_dim,
            'num_hidden_layers': num_hidden_layers,
            'total_parameters': model_data.count_parameters()
        },
        'training_config': {
            'num_epochs': num_epochs,
            'lr_adam': lr_adam,
            'lr_lbfgs': lr_lbfgs,
            'switch_epoch': switch_epoch,
            'lambda_bc': lambda_bc
        },
        'data_driven': {
            'final_loss': history_data['loss'][-1],
            'final_l2_error': history_data['l2_error'][-1],
            'l2_relative_error': metrics_data['l2_relative_error'],
            'linf_relative_error': metrics_data['linf_relative_error'],
            'mae': metrics_data['mae']
        },
        'pinn': {
            'final_loss': history_pinn['loss'][-1],
            'final_physics_loss': history_pinn['physics_loss'][-1],
            'final_bc_loss': history_pinn['bc_loss'][-1],
            'final_l2_error': history_pinn['l2_error'][-1],
            'l2_relative_error': metrics_pinn['l2_relative_error'],
            'linf_relative_error': metrics_pinn['linf_relative_error'],
            'mae': metrics_pinn['mae']
        }
    }
    
    # Save results to JSON
    with open(save_path / f"results_sample_{sample_idx}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n6. Results saved to {save_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"K = {K_value}, Sample = {sample_idx}")
    print("-" * 80)
    print(f"{'Method':<20} {'L2 Error':<15} {'L∞ Error':<15} {'MAE':<15}")
    print("-" * 80)
    print(f"{'Data-Driven':<20} {metrics_data['l2_relative_error']:<15.6e} "
          f"{metrics_data['linf_relative_error']:<15.6e} {metrics_data['mae']:<15.6e}")
    print(f"{'PINN':<20} {metrics_pinn['l2_relative_error']:<15.6e} "
          f"{metrics_pinn['linf_relative_error']:<15.6e} {metrics_pinn['mae']:<15.6e}")
    print("=" * 80)
    
    return results


def main():
    """Main function to run experiments."""
    parser = argparse.ArgumentParser(description='Train PINN and Data-Driven models')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Directory containing the dataset')
    parser.add_argument('--K', type=int, choices=[1, 4, 8, 16], required=True,
                       help='K value (complexity level)')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to use')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of hidden layers')
    parser.add_argument('--epochs', type=int, default=5000,
                       help='Number of training epochs')
    parser.add_argument('--lr_adam', type=float, default=1e-3,
                       help='Adam learning rate')
    parser.add_argument('--lr_lbfgs', type=float, default=1.0,
                       help='L-BFGS learning rate')
    parser.add_argument('--switch_epoch', type=int, default=3000,
                       help='Epoch to switch from Adam to L-BFGS')
    parser.add_argument('--lambda_bc', type=float, default=1.0,
                       help='Boundary condition weight for PINN')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_experiment(
        data_dir=args.data_dir,
        K_value=args.K,
        sample_idx=args.sample_idx,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_layers,
        num_epochs=args.epochs,
        lr_adam=args.lr_adam,
        lr_lbfgs=args.lr_lbfgs,
        switch_epoch=args.switch_epoch,
        lambda_bc=args.lambda_bc,
        device=args.device,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()