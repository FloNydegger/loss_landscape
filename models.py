"""
This module defines the MLP architecture used for both PINN and 
supervised learning approaches to solve the Poisson equation.
"""

import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for function approximation.
    
    Architecture:
    - Input: 2D coordinates (x, y)
    - Hidden layers: 3-4 layers with configurable width
    - Output: Scalar value u(x, y)
    - Activation: Tanh (smooth for automatic differentiation)
    """
    
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1, num_hidden_layers=4):
        """
        Initialize the MLP.
        
        Parameters:
        input_dim : int
            Input dimension (2 for x, y coordinates)
        hidden_dim : int
            Number of neurons in each hidden layer
        output_dim : int
            Output dimension (1 for scalar u)
        num_hidden_layers : int
            Number of hidden layers (default: 4)
        """
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input coordinates of shape (N, 2) where N is number of points
            
        Returns:
        --------
        u : torch.Tensor
            Predicted values of shape (N, 1)
        """
        return self.network(x)
    
    def count_parameters(self):
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_architecture_info(self):
        """Get a summary of the network architecture."""
        info = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_hidden_layers': self.num_hidden_layers,
            'total_parameters': self.count_parameters(),
            'activation': 'Tanh'
        }
        return info


def compute_laplacian(u, x, create_graph=True):
    """
    Compute the Laplacian ∇²u = ∂²u/∂x² + ∂²u/∂y² using automatic differentiation.
    
    This is used for the physics loss in PINN training.
    
    Parameters:
    -----------
    u : torch.Tensor
        Network output u(x,y) of shape (N, 1)
    x : torch.Tensor
        Input coordinates of shape (N, 2) with requires_grad=True
    create_graph : bool
        Whether to create computation graph for higher-order derivatives
        
    Returns:
    --------
    laplacian : torch.Tensor
        Laplacian ∇²u of shape (N, 1)
    """
    # First derivatives
    du_dx = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=create_graph,
        retain_graph=True
    )[0]
    
    # Split into x and y components
    du_dx_x = du_dx[:, 0:1]  # ∂u/∂x
    du_dx_y = du_dx[:, 1:2]  # ∂u/∂y
    
    # Second derivatives
    d2u_dx2 = torch.autograd.grad(
        outputs=du_dx_x,
        inputs=x,
        grad_outputs=torch.ones_like(du_dx_x),
        create_graph=create_graph,
        retain_graph=True
    )[0][:, 0:1]  # ∂²u/∂x²
    
    d2u_dy2 = torch.autograd.grad(
        outputs=du_dx_y,
        inputs=x,
        grad_outputs=torch.ones_like(du_dx_y),
        create_graph=create_graph,
        retain_graph=True
    )[0][:, 1:2]  # ∂²u/∂y²
    
    # Laplacian: ∇²u = ∂²u/∂x² + ∂²u/∂y²
    laplacian = d2u_dx2 + d2u_dy2
    
    return laplacian


def test_laplacian_computation():
    """
    Test the Laplacian computation with a known function.
    
    Test function: u(x,y) = sin(πx)sin(πy)
    Exact Laplacian: ∇²u = -2π²sin(πx)sin(πy)
    """
    print("Testing Laplacian computation...")
    print("-" * 50)
    
    # Create test points
    x = torch.linspace(0, 1, 10)
    y = torch.linspace(0, 1, 10)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
    coords.requires_grad = True
    
    # Test function: u = sin(πx)sin(πy)
    u = torch.sin(np.pi * coords[:, 0:1]) * torch.sin(np.pi * coords[:, 1:2])
    
    # Compute Laplacian using our function
    laplacian_computed = compute_laplacian(u, coords)
    
    # Exact Laplacian: -2π²sin(πx)sin(πy)
    laplacian_exact = -2 * np.pi**2 * torch.sin(np.pi * coords[:, 0:1]) * torch.sin(np.pi * coords[:, 1:2])
    
    # Compute error
    error = torch.abs(laplacian_computed - laplacian_exact).mean().item()
    
    print(f"Mean absolute error: {error:.2e}")
    print(f"✓ Laplacian computation {'PASSED' if error < 1e-5 else 'FAILED'}")
    print()


if __name__ == "__main__":
    # Test MLP architecture
    print("=" * 60)
    print("MLP Architecture Test")
    print("=" * 60)
    print()
    
    # Create model
    model = MLP(input_dim=2, hidden_dim=64, output_dim=1, num_hidden_layers=4)
    
    # Print architecture info
    info = model.get_architecture_info()
    print("Network Architecture:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()
    
    # Test forward pass
    batch_size = 100
    x_test = torch.randn(batch_size, 2)
    u_pred = model(x_test)
    
    print(f"Test forward pass:")
    print(f"  Input shape: {x_test.shape}")
    print(f"  Output shape: {u_pred.shape}")
    print(f"  Output range: [{u_pred.min():.4f}, {u_pred.max():.4f}]")
    print()
    
    # Test Laplacian computation
    test_laplacian_computation()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)