"""
Generate and Train on One Representative Instance

This script:
1. Generates ONE specific source term f with chosen complexity K
2. Trains both PINN and Data-Driven methods on this same instance
3. Allows you to compare methods on the exact same problem

This is useful when you want to use a "representative instance" across
different experiments while keeping the source term f constant.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Add parent directory to path if running from project root
sys.path.append('.')

from generate_data import PoissonDataGenerator
from models import MLP
from train import train_data_driven, train_pinn, evaluate_model
from model_checkpoint import save_model_checkpoint


def generate_single_instance(K, N=64, seed=42, save_dir='single_instance'):
    """Generate one representative instance with specified complexity K."""
    print(f"\nGenerating representative instance with K={K}, seed={seed}")
    print("-" * 70)
    
    generator = PoissonDataGenerator(N=N, r=0.5)
    f, u, a_ij = generator.generate_sample(K, seed=seed)
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    instance = {
        'K': K,
        'N': N,
        'seed': seed,
        'f': f,
        'u': u,
        'a_ij': a_ij
    }
    
    np.savez(
        save_path / f'instance_K{K}_seed{seed}.npz',
        K=K, N=N, seed=seed, f=f, u=u, a_ij=a_ij
    )
    
    metadata = {
        'K': int(K),
        'N': int(N),
        'seed': int(seed),
        'f_range': [float(f.min()), float(f.max())],
        'u_range': [float(u.min()), float(u.max())]
    }
    
    with open(save_path / f'instance_K{K}_seed{seed}_metadata.json', 'w') as file:
        json.dump(metadata, file, indent=2)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(f, cmap='RdBu_r', origin='lower', extent=[0, 1, 0, 1])
    axes[0].set_title(f'Source f (K={K})', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(u, cmap='RdBu_r', origin='lower', extent=[0, 1, 0, 1])
    axes[1].set_title(f'Ground Truth u (K={K})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(save_path / f'instance_K{K}_seed{seed}_visualization.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Instance saved to: {save_path}")
    
    return instance


def load_single_instance(K, seed=42, load_dir='single_instance'):
    """Load a previously generated instance."""
    load_path = Path(load_dir)
    data = np.load(load_path / f'instance_K{K}_seed{seed}.npz')
    
    instance = {
        'K': int(data['K']),
        'N': int(data['N']),
        'seed': int(data['seed']),
        'f': data['f'],
        'u': data['u'],
        'a_ij': data['a_ij']
    }
    
    print(f"\nLoaded instance: K={instance['K']}, N={instance['N']}, seed={instance['seed']}")
    return instance


def grid_to_coords(N, domain=[0, 1]):
    """Convert grid to coordinates."""
    x = np.linspace(domain[0], domain[1], N)
    y = np.linspace(domain[0], domain[1], N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    coords = np.stack([X.flatten(), Y.flatten()], axis=1)
    return coords


def train_on_instance(instance, 
                     hidden_dim=64, num_hidden_layers=4,
                     num_epochs=5000, lr_adam=1e-3, lr_lbfgs=1.0, 
                     switch_epoch=3000, lambda_bc=1.0,
                     device='cpu', save_dir='results_single_instance',
                     save_models=True):  # NEW PARAMETER
    """
    Train both PINN and Data-Driven methods on a single instance.
    NOW WITH MODEL SAVING!
    """
    K = instance['K']
    N = instance['N']
    seed = instance['seed']
    
    print("\n" + "=" * 80)
    print(f"Training on Single Instance: K={K}, seed={seed}")
    print("=" * 80)
    
    save_path = Path(save_dir) / f"K{K}_seed{seed}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    print(f"\n1. Preparing data...")
    coords = grid_to_coords(N)
    coords = torch.tensor(coords, dtype=torch.float32, device=device)
    
    f_values = torch.tensor(instance['f'].flatten(), dtype=torch.float32, device=device).unsqueeze(-1)
    u_exact = torch.tensor(instance['u'].flatten(), dtype=torch.float32, device=device).unsqueeze(-1)
    
    print(f"   Grid size: {N} x {N}")
    print(f"   Total points: {coords.shape[0]}")
    
    # Train Data-Driven model
    print(f"\n2. Training Data-Driven model...")
    model_data = MLP(input_dim=2, hidden_dim=hidden_dim, 
                     output_dim=1, num_hidden_layers=num_hidden_layers).to(device)
    print(f"   Model parameters: {model_data.count_parameters()}")
    
    history_data = train_data_driven(
        model_data, coords, f_values, u_exact,
        num_epochs=num_epochs, lr_adam=lr_adam, lr_lbfgs=lr_lbfgs,
        switch_epoch=switch_epoch, device=device, verbose=True
    )
    
    metrics_data = evaluate_model(model_data, coords, u_exact, device=device)
    print(f"   Final L2 relative error: {metrics_data['l2_relative_error']:.6e}")
    
    # SAVE DATA-DRIVEN MODEL
    if save_models:
        print(f"\n   Saving Data-Driven model...")
        config_data = {
            'method': 'data_driven',
            'K': K,
            'N': N,
            'seed': seed,
            'hidden_dim': hidden_dim,
            'num_hidden_layers': num_hidden_layers,
            'num_epochs': num_epochs,
            'lr_adam': lr_adam,
            'lr_lbfgs': lr_lbfgs,
            'switch_epoch': switch_epoch
        }
        save_model_checkpoint(model_data, history_data, metrics_data, config_data, save_path)
    
    # Train PINN model
    print(f"\n3. Training PINN model...")
    model_pinn = MLP(input_dim=2, hidden_dim=hidden_dim,
                     output_dim=1, num_hidden_layers=num_hidden_layers).to(device)
    print(f"   Model parameters: {model_pinn.count_parameters()}")
    
    history_pinn = train_pinn(
        model_pinn, coords, f_values, u_exact,
        num_epochs=num_epochs, lr_adam=lr_adam, lr_lbfgs=lr_lbfgs,
        switch_epoch=switch_epoch, lambda_bc=lambda_bc, device=device, verbose=True
    )
    
    metrics_pinn = evaluate_model(model_pinn, coords, u_exact, device=device)
    print(f"   Final L2 relative error: {metrics_pinn['l2_relative_error']:.6e}")
    
    # SAVE PINN MODEL
    if save_models:
        print(f"\n   💾 Saving PINN model...")
        config_pinn = {
            'method': 'pinn',
            'K': K,
            'N': N,
            'seed': seed,
            'hidden_dim': hidden_dim,
            'num_hidden_layers': num_hidden_layers,
            'num_epochs': num_epochs,
            'lr_adam': lr_adam,
            'lr_lbfgs': lr_lbfgs,
            'switch_epoch': switch_epoch,
            'lambda_bc': lambda_bc
        }
        save_model_checkpoint(model_pinn, history_pinn, metrics_pinn, config_pinn, save_path)
    
    # Generate plots (same as before)
    print(f"\n4. Generating visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].semilogy(history_data['epoch'], history_data['loss'], 
                     label='Data-Driven', linewidth=2, alpha=0.8)
    axes[0].semilogy(history_pinn['epoch'], history_pinn['loss'], 
                     label='PINN', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title(f'Training Loss (K={K})', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogy(history_data['epoch'], history_data['l2_error'], 
                     label='Data-Driven', linewidth=2, alpha=0.8)
    axes[1].semilogy(history_pinn['epoch'], history_pinn['l2_error'], 
                     label='PINN', linewidth=2, alpha=0.8)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('L2 Relative Error', fontsize=12)
    axes[1].set_title(f'L2 Relative Error (K={K})', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Predictions plot
    u_exact_grid = u_exact.cpu().numpy().reshape(N, N)
    u_pred_data_grid = metrics_data['u_pred'].reshape(N, N)
    u_pred_pinn_grid = metrics_pinn['u_pred'].reshape(N, N)
    
    error_data = np.abs(u_exact_grid - u_pred_data_grid)
    error_pinn = np.abs(u_exact_grid - u_pred_pinn_grid)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    vmin = min(u_exact_grid.min(), u_pred_data_grid.min(), u_pred_pinn_grid.min())
    vmax = max(u_exact_grid.max(), u_pred_data_grid.max(), u_pred_pinn_grid.max())
    
    im0 = axes[0, 0].imshow(u_exact_grid, cmap='RdBu_r', origin='lower', 
                            extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Ground Truth u', fontsize=12, fontweight='bold')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(u_pred_data_grid, cmap='RdBu_r', origin='lower',
                            extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Data-Driven Prediction', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(u_pred_pinn_grid, cmap='RdBu_r', origin='lower',
                            extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
    axes[0, 2].set_title('PINN Prediction', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 2])
    
    axes[1, 0].axis('off')
    
    error_max = max(error_data.max(), error_pinn.max())
    
    im3 = axes[1, 1].imshow(error_data, cmap='hot', origin='lower',
                            extent=[0, 1, 0, 1], vmin=0, vmax=error_max)
    axes[1, 1].set_title('Data-Driven Error', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 1])
    
    im4 = axes[1, 2].imshow(error_pinn, cmap='hot', origin='lower',
                            extent=[0, 1, 0, 1], vmin=0, vmax=error_max)
    axes[1, 2].set_title('PINN Error', fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 2])
    
    plt.suptitle(f'Predictions and Errors (K={K})', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path / 'predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved plots to {save_path}")
    
    # Save results
    results = {
        'instance': {'K': K, 'N': N, 'seed': seed},
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
    
    with open(save_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n5. Results saved to {save_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Instance: K={K}, seed={seed}")
    print("-" * 80)
    print(f"{'Method':<20} {'L2 Error':<15} {'L∞ Error':<15} {'MAE':<15}")
    print("-" * 80)
    print(f"{'Data-Driven':<20} {metrics_data['l2_relative_error']:<15.6e} "
          f"{metrics_data['linf_relative_error']:<15.6e} {metrics_data['mae']:<15.6e}")
    print(f"{'PINN':<20} {metrics_pinn['l2_relative_error']:<15.6e} "
          f"{metrics_pinn['linf_relative_error']:<15.6e} {metrics_pinn['mae']:<15.6e}")
    print("=" * 80)
    
    return results, model_data, model_pinn  # Also return the models


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate and train on single instance (with model saving)')
    parser.add_argument('--K', type=int, required=True, choices=[1, 4, 8, 16],
                       help='Complexity level')
    parser.add_argument('--seed', type=int, default=11,  # Changed default to 11 as per your files
                       help='Random seed for instance generation')
    parser.add_argument('--N', type=int, default=64,
                       help='Grid size')
    parser.add_argument('--generate', action='store_true',
                       help='Generate new instance (default: load existing)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of hidden layers')
    parser.add_argument('--epochs', type=int, default=5000,
                       help='Number of training epochs')
    parser.add_argument('--switch_epoch', type=int, default=3000,
                       help='Epoch to switch optimizers')
    parser.add_argument('--lambda_bc', type=float, default=1.0,
                       help='PINN boundary condition weight')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save model checkpoints')
    
    args = parser.parse_args()
    
    # Generate or load instance
    if args.generate:
        instance = generate_single_instance(K=args.K, N=args.N, seed=args.seed)
    else:
        try:
            instance = load_single_instance(K=args.K, seed=args.seed)
        except FileNotFoundError:
            print(f"Instance not found. Generating new instance...")
            instance = generate_single_instance(K=args.K, N=args.N, seed=args.seed)
    
    # Train on instance (with model saving enabled by default)
    results, model_data, model_pinn = train_on_instance(
        instance,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_layers,
        num_epochs=args.epochs,
        switch_epoch=args.switch_epoch,
        lambda_bc=args.lambda_bc,
        device=args.device,
        save_models=not args.no_save  # Save unless --no_save is specified
    )
    
    print("\nTraining complete! Models saved and ready for Task 3 (loss landscape visualization).")


if __name__ == "__main__":
    main()
