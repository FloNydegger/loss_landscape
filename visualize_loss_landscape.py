"""
Loss Landscape Visualization

This module implements the visualization techniques from Li et al. (2018)
"Visualizing the Loss Landscape of Neural Nets" to compare PINN and
Data-Driven approaches across different complexity levels K.

Key Features:
- Filter-normalized random directions (Section 4 of Li et al.)
- 2D contour plots and 3D surface plots
- Side-by-side comparison of methods and K values
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import json
from tqdm import tqdm

from models import MLP, compute_laplacian
from model_checkpoint import load_model_checkpoint


def generate_random_direction(model, device='cpu'):
    """
    Generate a random Gaussian direction vector with the same structure as model parameters.
    
    Parameters:
    model : nn.Module
        Model to match parameter structure
    device : str
        Device to create tensor on
        
    Returns:
    direction : list of torch.Tensor
        Random direction vector matching model structure
    """
    direction = []
    for param in model.parameters():
        direction.append(torch.randn_like(param, device=device))
    return direction


def normalize_direction_filter_wise(direction, model):
    """
    Normalize direction using filter-wise normalization.    
    Parameters:
    direction : list of torch.Tensor
        Random direction to normalize
    model : nn.Module
        Model whose parameter norms to match
        
    Returns:
    --------
    normalized_direction : list of torch.Tensor
        Filter-normalized direction
    """
    normalized_direction = []
    
    for d, param in zip(direction, model.parameters()):
        # For each parameter tensor
        if len(param.shape) >= 2:  # Weight matrices (Conv or FC layers)
            # Normalize each filter (or row for FC layers)
            d_normalized = torch.zeros_like(d)
            
            if len(param.shape) == 2:  # Fully connected layer
                # Each row is a "filter"
                for i in range(param.shape[0]):
                    param_norm = torch.norm(param[i])
                    d_norm = torch.norm(d[i])
                    if d_norm > 0:
                        d_normalized[i] = (d[i] / d_norm) * param_norm
                    else:
                        d_normalized[i] = d[i]
            
            elif len(param.shape) == 4:  # Convolutional layer
                # Each filter is indexed by the first dimension
                for i in range(param.shape[0]):
                    param_norm = torch.norm(param[i])
                    d_norm = torch.norm(d[i])
                    if d_norm > 0:
                        d_normalized[i] = (d[i] / d_norm) * param_norm
                    else:
                        d_normalized[i] = d[i]
            else:
                # For other shapes, just copy
                d_normalized = d
            
            normalized_direction.append(d_normalized)
        else:
            # For biases, just use the unnormalized direction
            normalized_direction.append(d)
    
    return normalized_direction


def set_weights_from_direction(model, theta_star, direction, alpha):
    """
    Parameters:
    model : nn.Module
        Model to modify
    theta_star : list of torch.Tensor
        Base parameters (converged solution)
    direction : list of torch.Tensor
        Direction vector
    alpha : float
        Step size
    """
    with torch.no_grad():
        for param, theta, d in zip(model.parameters(), theta_star, direction):
            param.data = theta + alpha * d


def compute_loss_2d_grid(model, coords, f_values, u_exact, 
                         theta_star, dir1, dir2,
                         alpha_range, beta_range, 
                         method='pinn', lambda_bc=1.0,
                         device='cpu'):
    """
    Compute loss values on a 2D grid
    
    Parameters:
    model : nn.Module
        Neural network model
    coords : torch.Tensor
        Input coordinates
    f_values : torch.Tensor
        Source term values
    u_exact : torch.Tensor
        Exact solution (for Data-Driven loss)
    theta_star : list of torch.Tensor
        Converged parameters θ*
    dir1, dir2 : list of torch.Tensor
        Two direction vectors (δ, η)
    alpha_range, beta_range : np.ndarray
        Grid of α and β values
    method : str
        'pinn' or 'data_driven'
    lambda_bc : float
        Boundary condition weight (for PINN)
    device : str
        Device to compute on
        
    Returns:
    loss_grid : np.ndarray
        2D array of loss values
    """
    n_alpha = len(alpha_range)
    n_beta = len(beta_range)
    loss_grid = np.zeros((n_beta, n_alpha))
    
    # Identify boundary points for PINN
    if method == 'pinn':
        eps = 1e-6
        boundary_mask = (
            (coords[:, 0] < eps) | (coords[:, 0] > 1 - eps) |
            (coords[:, 1] < eps) | (coords[:, 1] > 1 - eps)
        )
        boundary_indices = torch.where(boundary_mask)[0]
    
    # Progress bar
    total_iterations = n_alpha * n_beta
    pbar = tqdm(total=total_iterations, desc=f"Computing loss grid ({method})")
    
    for i, alpha in enumerate(alpha_range):
        for j, beta in enumerate(beta_range):
            # Set weights: θ = θ* + α·δ + β·η
            with torch.no_grad():
                for param, theta, d1, d2 in zip(model.parameters(), theta_star, dir1, dir2):
                    param.data = theta + alpha * d1 + beta * d2
            
            # Compute loss
            model.eval()
            
            if method == 'pinn':
                # PINN loss - need gradients for Laplacian
                coords_grad = coords.detach().clone().requires_grad_(True)
                
                # Enable gradient computation for this forward pass
                u_pred = model(coords_grad)
                
                # Physics loss
                laplacian = compute_laplacian(u_pred, coords_grad)
                physics_residual = -laplacian - f_values
                physics_loss = torch.mean(physics_residual ** 2)
                
                # Boundary condition loss
                bc_loss = torch.mean(u_pred[boundary_indices] ** 2)
                
                # Total loss
                loss = physics_loss + lambda_bc * bc_loss
                
                # Detach for storage
                loss = loss.detach()
                    
            elif method == 'data_driven':
                # Data-Driven loss - no gradients needed
                with torch.no_grad():
                    u_pred = model(coords)
                    loss = torch.mean((u_pred - u_exact) ** 2)
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            loss_grid[j, i] = loss.item()
            pbar.update(1)
    
    pbar.close()
    return loss_grid


def plot_loss_landscape_2d(alpha_range, beta_range, loss_grid, 
                           method, K, test_error,
                           save_path=None, show_contours=True):
    """
    Create a 2D contour plot of the loss landscape.
    
    Parameters:
    alpha_range, beta_range : np.ndarray
        Grid ranges
    loss_grid : np.ndarray
        Loss values on the grid
    method : str
        'pinn' or 'data_driven'
    K : int
        Complexity level
    test_error : float
        Test error of the model
    save_path : Path or str, optional
        Where to save the plot
    show_contours : bool
        Whether to show contour lines
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create meshgrid
    Alpha, Beta = np.meshgrid(alpha_range, beta_range)
    
    # Contour plot
    if show_contours:
        # Generate contour levels (log scale for better visualization)
        min_loss = loss_grid.min()
        max_loss = np.percentile(loss_grid, 99)  # Cap at 99th percentile to avoid extreme outliers
        
        # Create levels
        levels = np.logspace(np.log10(min_loss + 0.001), np.log10(max_loss), 20)
        
        # Filled contours
        contourf = ax.contourf(Alpha, Beta, loss_grid, levels=levels, 
                               cmap='RdYlBu_r', alpha=0.8)
        
        # Contour lines
        contour = ax.contour(Alpha, Beta, loss_grid, levels=levels, 
                            colors='black', alpha=0.3, linewidths=0.5)
        
        # Colorbar
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Loss', fontsize=12)
    else:
        # Just color map
        im = ax.imshow(loss_grid, extent=[alpha_range[0], alpha_range[-1], 
                                          beta_range[0], beta_range[-1]],
                      origin='lower', cmap='RdYlBu_r', aspect='auto')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Loss', fontsize=12)
    
    # Mark the center (θ*)
    ax.plot(0, 0, 'r*', markersize=15, label='θ* (converged solution)', zorder=10)
    
    # Labels
    ax.set_xlabel('α (direction δ)', fontsize=12)
    ax.set_ylabel('β (direction η)', fontsize=12)
    
    method_name = "PINN" if method == 'pinn' else "Data-Driven"
    ax.set_title(f'{method_name} Loss Landscape (K={K})\n'
                f'Test L2 Error: {test_error:.6f}', 
                fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_loss_landscape_3d(alpha_range, beta_range, loss_grid,
                          method, K, test_error,
                          save_path=None):
    """
    Create a 3D surface plot of the loss landscape.
    
    Parameters:
    alpha_range, beta_range : np.ndarray
        Grid ranges
    loss_grid : np.ndarray
        Loss values on the grid
    method : str
        'pinn' or 'data_driven'
    K : int
        Complexity level
    test_error : float
        Test error of the model
    save_path : Path or str, optional
        Where to save the plot
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    Alpha, Beta = np.meshgrid(alpha_range, beta_range)
    
    # Cap extreme values for better visualization
    loss_plot = np.clip(loss_grid, loss_grid.min(), np.percentile(loss_grid, 95))
    
    # Surface plot
    surf = ax.plot_surface(Alpha, Beta, loss_plot, cmap='RdYlBu_r',
                          alpha=0.9, edgecolor='none', antialiased=True)
    
    # Mark the center (θ*)
    ax.scatter([0], [0], [loss_grid[len(beta_range)//2, len(alpha_range)//2]], 
              color='red', s=100, marker='*', label='θ* (converged)', zorder=10)
    
    # Labels
    ax.set_xlabel('α (direction δ)', fontsize=11)
    ax.set_ylabel('β (direction η)', fontsize=11)
    ax.set_zlabel('Loss', fontsize=11)
    
    method_name = "PINN" if method == 'pinn' else "Data-Driven"
    ax.set_title(f'{method_name} Loss Landscape (K={K})\n'
                f'Test L2 Error: {test_error:.6f}', 
                fontsize=13, fontweight='bold')
    
    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    ax.legend(fontsize=10)
    
    # Better viewing angle
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def visualize_single_landscape(checkpoint_path, instance_path,
                               method, K, 
                               alpha_min=-1, alpha_max=1, n_alpha=51,
                               beta_min=-1, beta_max=1, n_beta=51,
                               device='cpu', save_dir='task3_results'):
    """
    Visualize the loss landscape for a single trained model.
    
    Parameters:
    checkpoint_path : str or Path
        Path to model checkpoint
    instance_path : str or Path
        Path to instance data (.npz file)
    method : str
        'pinn' or 'data_driven'
    K : int
        Complexity level
    alpha_min, alpha_max : float
        Range for α direction
    n_alpha : int
        Number of grid points in α direction
    beta_min, beta_max : float
        Range for β direction
    n_beta : int
        Number of grid points in β direction
    device : str
        Device to use
    save_dir : str or Path
        Directory to save results
        
    Returns:
    results : dict
        Dictionary containing loss grid and metadata
    """
    print(f"\n{'='*70}")
    print(f"Visualizing Loss Landscape: {method.upper()}, K={K}")
    print(f"{'='*70}")
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Load checkpoint
    checkpoint = load_model_checkpoint(checkpoint_path, device=device)
    
    # Create model
    model_config = checkpoint['model_config']
    model = MLP(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        output_dim=model_config['output_dim'],
        num_hidden_layers=model_config['num_hidden_layers']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get θ* (converged parameters)
    theta_star = [param.data.clone() for param in model.parameters()]
    
    # Load instance data
    instance_data = np.load(instance_path)
    
    # Prepare data
    N = int(instance_data['N'])
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    coords = np.stack([X.flatten(), Y.flatten()], axis=1)
    coords = torch.tensor(coords, dtype=torch.float32, device=device)
    
    f_values = torch.tensor(instance_data['f'].flatten(), dtype=torch.float32, device=device).unsqueeze(-1)
    u_exact = torch.tensor(instance_data['u'].flatten(), dtype=torch.float32, device=device).unsqueeze(-1)
    
    print(f"\n1. Generating random directions...")
    
    # Generate two random directions
    dir1 = generate_random_direction(model, device=device)
    dir2 = generate_random_direction(model, device=device)
    
    print(f"2. Applying filter normalization (Li et al. 2018, Section 4)...")
    
    # Apply filter normalization (CRITICAL!)
    dir1_normalized = normalize_direction_filter_wise(dir1, model)
    dir2_normalized = normalize_direction_filter_wise(dir2, model)
    
    print(f"3. Computing loss on {n_alpha}×{n_beta} grid...")
    print(f"   α range: [{alpha_min}, {alpha_max}]")
    print(f"   β range: [{beta_min}, {beta_max}]")
    
    # Create grid
    alpha_range = np.linspace(alpha_min, alpha_max, n_alpha)
    beta_range = np.linspace(beta_min, beta_max, n_beta)
    
    # Get lambda_bc if PINN
    lambda_bc = checkpoint['training_config'].get('lambda_bc', 1.0) if method == 'pinn' else 1.0
    
    # Compute loss grid
    loss_grid = compute_loss_2d_grid(
        model, coords, f_values, u_exact,
        theta_star, dir1_normalized, dir2_normalized,
        alpha_range, beta_range,
        method=method, lambda_bc=lambda_bc,
        device=device
    )
    
    # Restore original weights
    set_weights_from_direction(model, theta_star, dir1_normalized, 0)
    
    print(f"\n4. Creating visualizations...")
    
    # Get test error
    test_error = checkpoint['final_metrics']['l2_relative_error']
    
    # 2D contour plot
    contour_path = save_dir / f"{method}_K{K}_contour.png"
    plot_loss_landscape_2d(
        alpha_range, beta_range, loss_grid,
        method, K, test_error,
        save_path=contour_path
    )
    
    # 3D surface plot
    surface_path = save_dir / f"{method}_K{K}_surface.png"
    plot_loss_landscape_3d(
        alpha_range, beta_range, loss_grid,
        method, K, test_error,
        save_path=surface_path
    )
    
    # Save loss grid data
    data_path = save_dir / f"{method}_K{K}_data.npz"
    np.savez(
        data_path,
        loss_grid=loss_grid,
        alpha_range=alpha_range,
        beta_range=beta_range,
        test_error=test_error,
        method=method,
        K=K
    )
    print(f"Saved: {data_path}")
    
    # Statistics
    print(f"\n{'='*70}")
    print(f"STATISTICS")
    print(f"{'='*70}")
    print(f"Min loss:        {loss_grid.min():.6f}")
    print(f"Max loss:        {loss_grid.max():.6f}")
    print(f"Loss at θ*:      {loss_grid[n_beta//2, n_alpha//2]:.6f}")
    print(f"Test L2 error:   {test_error:.6f}")
    print(f"{'='*70}")
    
    results = {
        'loss_grid': loss_grid,
        'alpha_range': alpha_range,
        'beta_range': beta_range,
        'test_error': test_error,
        'method': method,
        'K': K,
        'min_loss': float(loss_grid.min()),
        'max_loss': float(loss_grid.max()),
        'center_loss': float(loss_grid[n_beta//2, n_alpha//2])
    }
    
    return results


if __name__ == "__main__":
    print("Loss Landscape Visualization Module")
    print("=" * 70)
    print("Use run_task3.py to execute the full visualization workflow.")