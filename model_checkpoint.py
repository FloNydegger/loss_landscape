"""
Model Checkpoint Utilities for Loss Landscape Visualization

This module provides utilities to save and load trained models
along with their training state for Task 3 (loss landscape visualization).
"""

import torch
import json
from pathlib import Path
import numpy as np


def save_model_checkpoint(model, history, metrics, config, save_path):
    """
    Save a complete model checkpoint including:
    - Model state dict (parameters θ*)
    - Training history
    - Final metrics
    - Configuration
    
    Parameters:
    -----------
    model : nn.Module
        Trained model
    history : dict
        Training history from train function
    metrics : dict
        Evaluation metrics from evaluate_model
    config : dict
        Configuration dictionary containing:
        - method: 'pinn' or 'data_driven'
        - K: complexity level
        - N: grid size
        - seed: random seed
        - hidden_dim: model hidden dimension
        - num_hidden_layers: number of hidden layers
        - etc.
    save_path : str or Path
        Directory to save checkpoint
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint filename
    method = config['method']
    K = config['K']
    seed = config.get('seed', 42)
    checkpoint_name = f"{method}_K{K}_seed{seed}_checkpoint.pt"
    
    # Save model state dict
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': 2,
            'hidden_dim': config['hidden_dim'],
            'output_dim': 1,
            'num_hidden_layers': config['num_hidden_layers']
        },
        'training_config': config,
        'final_metrics': {
            'l2_relative_error': metrics['l2_relative_error'],
            'linf_relative_error': metrics['linf_relative_error'],
            'mae': metrics['mae']
        },
        'final_loss': history['loss'][-1],
        'final_epoch': history['epoch'][-1]
    }
    
    # Add method-specific information
    if method == 'pinn':
        checkpoint['final_physics_loss'] = history['physics_loss'][-1]
        checkpoint['final_bc_loss'] = history['bc_loss'][-1]
        checkpoint['lambda_bc'] = config.get('lambda_bc', 1.0)
    
    # Save checkpoint
    torch.save(checkpoint, save_path / checkpoint_name)
    
    # Save training history separately (easier to load for plotting)
    history_clean = {k: v for k, v in history.items()}
    np.savez(
        save_path / f"{method}_K{K}_seed{seed}_history.npz",
        **history_clean
    )
    
    # Save readable summary
    summary = {
        'method': method,
        'K': K,
        'seed': seed,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'final_l2_error': float(metrics['l2_relative_error']),
        'final_loss': float(history['loss'][-1]),
        'total_epochs': int(history['epoch'][-1]) + 1
    }
    
    with open(save_path / f"{method}_K{K}_seed{seed}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Checkpoint saved: {save_path / checkpoint_name}")
    print(f"  - Model state dict")
    print(f"  - Training history")
    print(f"  - Final metrics")
    
    return save_path / checkpoint_name


def load_model_checkpoint(checkpoint_path, model=None, device='cpu'):
    """
    Load a model checkpoint.
    
    Parameters:
    -----------
    checkpoint_path : str or Path
        Path to checkpoint file
    model : nn.Module, optional
        If provided, load state dict into this model
        If None, return checkpoint dict only
    device : str
        Device to load model to
        
    Returns:
    --------
    If model provided:
        model : loaded model
        checkpoint : checkpoint dictionary
    If model is None:
        checkpoint : checkpoint dictionary only
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model is not None:
        # Load state dict into model
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"✓ Model loaded from: {checkpoint_path}")
        print(f"  Method: {checkpoint['training_config']['method']}")
        print(f"  K: {checkpoint['training_config']['K']}")
        print(f"  Final L2 error: {checkpoint['final_metrics']['l2_relative_error']:.6e}")
        
        return model, checkpoint
    else:
        return checkpoint


def get_model_parameters_as_vector(model):
    """
    Extract all model parameters as a single flat vector.
    Useful for loss landscape visualization.
    
    Parameters:
    -----------
    model : nn.Module
        Model to extract parameters from
        
    Returns:
    --------
    params : torch.Tensor
        Flat vector of all parameters
    """
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    return torch.cat(params)


def set_model_parameters_from_vector(model, param_vector):
    """
    Set model parameters from a flat vector.
    Used for perturbing parameters during loss landscape visualization.
    
    Parameters:
    -----------
    model : nn.Module
        Model to set parameters for
    param_vector : torch.Tensor
        Flat vector of parameters
    """
    offset = 0
    for param in model.parameters():
        param_length = param.numel()
        param.data = param_vector[offset:offset + param_length].view(param.shape)
        offset += param_length


def find_checkpoints(search_dir, method=None, K=None):
    """
    Find all checkpoints in a directory.
    
    Parameters:
    -----------
    search_dir : str or Path
        Directory to search
    method : str, optional
        Filter by method ('pinn' or 'data_driven')
    K : int, optional
        Filter by complexity level
        
    Returns:
    --------
    checkpoints : list
        List of checkpoint paths
    """
    search_dir = Path(search_dir)
    
    # Find all checkpoint files
    checkpoints = list(search_dir.rglob('*_checkpoint.pt'))
    
    # Filter by method if specified
    if method is not None:
        checkpoints = [c for c in checkpoints if method in c.name]
    
    # Filter by K if specified
    if K is not None:
        checkpoints = [c for c in checkpoints if f'K{K}' in c.name]
    
    return sorted(checkpoints)


if __name__ == "__main__":
    print("Model checkpoint utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - save_model_checkpoint()")
    print("  - load_model_checkpoint()")
    print("  - get_model_parameters_as_vector()")
    print("  - set_model_parameters_from_vector()")
    print("  - find_checkpoints()")