"""
Training Functions for PINN and Data-Driven Approaches

This module contains the training logic for both physics-informed
neural networks (PINN) and supervised data-driven approaches.
"""

import torch
import torch.nn as nn
import numpy as np
from models import compute_laplacian
import time


def train_data_driven(model, coords, f_values, u_exact, 
                     num_epochs=5000, lr_adam=1e-3, lr_lbfgs=1.0,
                     switch_epoch=3000, device='cpu', verbose=True):
    """
    Train the model using Data-Driven (supervised) loss.
    
    Parameters:
    model : nn.Module
        Neural network model
    coords : torch.Tensor
        Input coordinates (N, 2)
    f_values : torch.Tensor
        Source term values (N, 1) - not used in data-driven, but kept for consistency
    u_exact : torch.Tensor
        Exact solution values (N, 1) - supervision target
    num_epochs : int
        Total number of training epochs
    lr_adam : float
        Learning rate for Adam optimizer
    lr_lbfgs : float
        Learning rate for L-BFGS optimizer
    switch_epoch : int
        Epoch to switch from Adam to L-BFGS
    device : str
        Device to train on ('cpu' or 'cuda')
    verbose : bool
        Whether to print training progress
        
    Returns:
    history : dict
        Training history containing losses and metrics
    """
    model = model.to(device)
    coords = coords.to(device)
    u_exact = u_exact.to(device)
    
    # Training history
    history = {
        'epoch': [],
        'loss': [],
        'l2_error': [],
        'time': [],
        'optimizer': []
    }
    
    # Phase 1: Adam optimizer
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr_adam)
    
    if verbose:
        print("=" * 70)
        print("Data-Driven Training")
        print("=" * 70)
        print(f"Phase 1: Adam optimizer (epochs 0-{switch_epoch})")
        print(f"Phase 2: L-BFGS optimizer (epochs {switch_epoch}-{num_epochs})")
        print("-" * 70)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Switch to L-BFGS
        if epoch == switch_epoch:
            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=lr_lbfgs,
                max_iter=20,
                history_size=50,
                line_search_fn='strong_wolfe'
            )
            if verbose:
                print(f"\n{'='*70}")
                print(f"Switching to L-BFGS optimizer at epoch {epoch}")
                print(f"{'='*70}\n")
        
        # Use Adam for early epochs
        if epoch < switch_epoch:
            optimizer_adam.zero_grad()
            
            # Forward pass
            u_pred = model(coords)
            
            # Data-driven loss: MSE between prediction and exact solution
            loss = torch.mean((u_pred - u_exact) ** 2)
            
            # Backward pass
            loss.backward()
            optimizer_adam.step()
            
            optimizer_name = 'Adam'
        
        # Use L-BFGS for later epochs
        else:
            def closure():
                optimizer.zero_grad()
                u_pred = model(coords)
                loss = torch.mean((u_pred - u_exact) ** 2)
                loss.backward()
                return loss
            
            optimizer.step(closure)
            
            # Recompute loss for logging
            with torch.no_grad():
                u_pred = model(coords)
                loss = torch.mean((u_pred - u_exact) ** 2)
            
            optimizer_name = 'L-BFGS'
        
        # Compute metrics
        with torch.no_grad():
            u_pred = model(coords)
            
            # L2 relative error
            l2_error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)) / torch.sqrt(torch.mean(u_exact ** 2))
            
            # Log history
            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['l2_error'].append(l2_error.item())
            history['time'].append(time.time() - start_time)
            history['optimizer'].append(optimizer_name)
        
        # Print progress
        if verbose and (epoch % 500 == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.6e} | "
                  f"L2 Error: {l2_error.item():.6e} | Opt: {optimizer_name}")
    
    if verbose:
        print("=" * 70)
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        print(f"Final L2 relative error: {history['l2_error'][-1]:.6e}")
        print("=" * 70)
        print()
    
    return history


def train_pinn(model, coords, f_values, u_exact,
              num_epochs=5000, lr_adam=1e-3, lr_lbfgs=1.0,
              switch_epoch=3000, lambda_bc=1.0, device='cpu', verbose=True):
    """
    Train the model using Physics-Informed Neural Network (PINN) loss.
    
    Parameters:
    model : nn.Module
        Neural network model
    coords : torch.Tensor
        Input coordinates (N, 2)
    f_values : torch.Tensor
        Source term values (N, 1)
    u_exact : torch.Tensor
        Exact solution values (N, 1) - only used for computing error metrics
    num_epochs : int
        Total number of training epochs
    lr_adam : float
        Learning rate for Adam optimizer
    lr_lbfgs : float
        Learning rate for L-BFGS optimizer
    switch_epoch : int
        Epoch to switch from Adam to L-BFGS
    lambda_bc : float
        Weight for boundary condition loss
    device : str
        Device to train on ('cpu' or 'cuda')
    verbose : bool
        Whether to print training progress
        
    Returns:
    history : dict
        Training history containing losses and metrics
    """
    model = model.to(device)
    coords = coords.to(device).requires_grad_(True)
    f_values = f_values.to(device)
    u_exact = u_exact.to(device)
    
    # Identify boundary points (x=0, x=1, y=0, y=1)
    eps = 1e-6
    boundary_mask = (
        (coords[:, 0] < eps) | (coords[:, 0] > 1 - eps) |
        (coords[:, 1] < eps) | (coords[:, 1] > 1 - eps)
    )
    boundary_indices = torch.where(boundary_mask)[0]
    interior_indices = torch.where(~boundary_mask)[0]
    
    # Training history
    history = {
        'epoch': [],
        'loss': [],
        'physics_loss': [],
        'bc_loss': [],
        'l2_error': [],
        'time': [],
        'optimizer': []
    }
    
    # Phase 1: Adam optimizer
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr_adam)
    
    if verbose:
        print("=" * 70)
        print("PINN Training")
        print("=" * 70)
        print(f"Phase 1: Adam optimizer (epochs 0-{switch_epoch})")
        print(f"Phase 2: L-BFGS optimizer (epochs {switch_epoch}-{num_epochs})")
        print(f"Boundary points: {len(boundary_indices)}")
        print(f"Interior points: {len(interior_indices)}")
        print(f"λ_BC: {lambda_bc}")
        print("-" * 70)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Switch to L-BFGS
        if epoch == switch_epoch:
            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=lr_lbfgs,
                max_iter=20,
                history_size=50,
                line_search_fn='strong_wolfe'
            )
            if verbose:
                print(f"\n{'='*70}")
                print(f"Switching to L-BFGS optimizer at epoch {epoch}")
                print(f"{'='*70}\n")
        
        # Use Adam for early epochs
        if epoch < switch_epoch:
            optimizer_adam.zero_grad()
            
            # Forward pass
            u_pred = model(coords)
            
            # Physics loss: |−∇²u − f|²
            laplacian = compute_laplacian(u_pred, coords)
            physics_residual = -laplacian - f_values
            physics_loss = torch.mean(physics_residual ** 2)
            
            # Boundary condition loss: |u|_boundary = 0
            bc_loss = torch.mean(u_pred[boundary_indices] ** 2)
            
            # Total loss
            loss = physics_loss + lambda_bc * bc_loss
            
            # Backward pass
            loss.backward()
            optimizer_adam.step()
            
            optimizer_name = 'Adam'
        
        # Use L-BFGS for later epochs
        else:
            def closure():
                optimizer.zero_grad()
                u_pred = model(coords)
                
                # Physics loss
                laplacian = compute_laplacian(u_pred, coords)
                physics_residual = -laplacian - f_values
                physics_loss = torch.mean(physics_residual ** 2)
                
                # Boundary condition loss
                bc_loss = torch.mean(u_pred[boundary_indices] ** 2)
                
                # Total loss
                loss = physics_loss + lambda_bc * bc_loss
                loss.backward()
                return loss
            
            optimizer.step(closure)
            
            # Recompute losses for logging
            with torch.no_grad():
                u_pred = model(coords)
                laplacian = compute_laplacian(u_pred, coords, create_graph=False)
                physics_residual = -laplacian - f_values
                physics_loss = torch.mean(physics_residual ** 2)
                bc_loss = torch.mean(u_pred[boundary_indices] ** 2)
                loss = physics_loss + lambda_bc * bc_loss
            
            optimizer_name = 'L-BFGS'
        
        # Compute metrics
        with torch.no_grad():
            u_pred = model(coords)
            
            # L2 relative error (compared to exact solution)
            l2_error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)) / torch.sqrt(torch.mean(u_exact ** 2))
            
            # Log history
            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['physics_loss'].append(physics_loss.item())
            history['bc_loss'].append(bc_loss.item())
            history['l2_error'].append(l2_error.item())
            history['time'].append(time.time() - start_time)
            history['optimizer'].append(optimizer_name)
        
        # Print progress
        if verbose and (epoch % 500 == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.6e} | "
                  f"Physics: {physics_loss.item():.6e} | BC: {bc_loss.item():.6e} | "
                  f"L2 Error: {l2_error.item():.6e} | Opt: {optimizer_name}")
    
    if verbose:
        print("=" * 70)
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        print(f"Final L2 relative error: {history['l2_error'][-1]:.6e}")
        print("=" * 70)
        print()
    
    return history


def evaluate_model(model, coords, u_exact, device='cpu'):
    """
    Evaluate the trained model and compute error metrics.
    
    Parameters:
    model : nn.Module
        Trained neural network model
    coords : torch.Tensor
        Input coordinates (N, 2)
    u_exact : torch.Tensor
        Exact solution values (N, 1)
    device : str
        Device to evaluate on
        
    Returns:
    metrics : dict
        Dictionary containing various error metrics
    """
    model = model.to(device)
    model.eval()
    
    coords = coords.to(device)
    u_exact = u_exact.to(device)
    
    with torch.no_grad():
        u_pred = model(coords)
        
        # L2 relative error
        l2_error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)) / torch.sqrt(torch.mean(u_exact ** 2))
        
        # L∞ error
        linf_error = torch.max(torch.abs(u_pred - u_exact)) / torch.max(torch.abs(u_exact))
        
        # Mean absolute error
        mae = torch.mean(torch.abs(u_pred - u_exact))
        
        # Point-wise errors
        pointwise_errors = torch.abs(u_pred - u_exact).cpu().numpy()
        
        metrics = {
            'l2_relative_error': l2_error.item(),
            'linf_relative_error': linf_error.item(),
            'mae': mae.item(),
            'u_pred': u_pred.cpu().numpy(),
            'u_exact': u_exact.cpu().numpy(),
            'pointwise_errors': pointwise_errors
        }
    
    return metrics


if __name__ == "__main__":
    print("Training functions module loaded successfully!")
    print("\nAvailable functions:")
    print("  - train_data_driven()")
    print("  - train_pinn()")
    print("  - evaluate_model()")