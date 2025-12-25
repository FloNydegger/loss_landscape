"""
Data Loading Utility for Multiscale Poisson Dataset

This module provides utilities to load and work with the generated dataset.
"""

import numpy as np
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class PoissonDataset(Dataset):
    
    def __init__(self, data_dir, K_value, transform=None):
        """
        Parameters:
        data_dir : str or Path
            Root directory containing the data
        K_value : int
            Which K value to load (e.g., 1, 4, 8, 16)
        transform : callable, optional
            Optional transform to apply to samples
        """
        self.data_dir = Path(data_dir)
        self.K_value = K_value
        self.transform = transform
        
        # Load metadata
        with open(self.data_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Get list of sample files
        self.sample_dir = self.data_dir / f'K_{K_value}'
        self.sample_files = sorted(list(self.sample_dir.glob('sample_*.npz')))
        
        if len(self.sample_files) == 0:
            raise ValueError(f"No samples found for K={K_value} in {self.sample_dir}")
        
        print(f"Loaded {len(self.sample_files)} samples for K={K_value}")
    
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
        sample : dict
            Dictionary containing 'f', 'u', 'a_ij', and 'seed'
        """
        # Load sample
        data = np.load(self.sample_files[idx])
        
        sample = {
            'f': torch.from_numpy(data['f']).float(),
            'u': torch.from_numpy(data['u']).float(),
            'a_ij': torch.from_numpy(data['a_ij']).float(),
            'seed': int(data['seed'])
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def create_coordinate_grid(N, domain=[0, 1], device='cpu'):
    """
    Create coordinate grid for the domain.
    
    Parameters:
    N : int
        Grid size (N x N)
    domain : list
        Domain bounds [x_min, x_max]
    device : str
        Device to place tensor on
        
    Returns:
    coords : torch.Tensor
        Coordinate grid of shape (N*N, 2) with columns [x, y]
    """
    x = torch.linspace(domain[0], domain[1], N, device=device)
    y = torch.linspace(domain[0], domain[1], N, device=device)
    
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Flatten and stack
    coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    return coords


def flatten_fields(f, u):
    """
    Flatten 2D fields to 1D vectors.
    
    Parameters:
    f : torch.Tensor
        Source field of shape (N, N)
    u : torch.Tensor
        Solution field of shape (N, N)
        
    Returns:
    f_flat : torch.Tensor
        Flattened source field of shape (N*N,)
    u_flat : torch.Tensor
        Flattened solution field of shape (N*N,)
    """
    return f.flatten(), u.flatten()


def load_metadata(data_dir):
    """
    Load dataset metadata.
    
    Parameters:
    data_dir : str or Path
        Root directory containing the data
        
    Returns:
    metadata : dict
        Dataset metadata
    """
    with open(Path(data_dir) / 'metadata.json', 'r') as f:
        return json.load(f)


def get_dataloader(data_dir, K_value, batch_size=32, shuffle=True, num_workers=0):
    """
    Create a DataLoader for the dataset.
    
    Parameters:
    data_dir : str or Path
        Root directory containing the data
    K_value : int
        Which K value to load
    batch_size : int
        Batch size
    shuffle : bool
        Whether to shuffle the data
    num_workers : int
        Number of worker processes for data loading
        
    Returns:
    dataloader : DataLoader
        PyTorch DataLoader
    """
    dataset = PoissonDataset(data_dir, K_value)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader


# Example usage
if __name__ == "__main__":
    print("Testing data loading utilities...")
    
    # Load metadata
    metadata = load_metadata('data')
    print("\nDataset Metadata:")
    print(f"  Grid size: {metadata['N']} x {metadata['N']}")
    print(f"  r parameter: {metadata['r']}")
    print(f"  K values: {metadata['K_values']}")
    print(f"  Samples per K: {metadata['num_samples_per_K']}")
    
    # Test loading a dataset
    K = 16
    dataset = PoissonDataset('data', K_value=K)
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample shapes for K={K}:")
    print(f"  f: {sample['f'].shape}")
    print(f"  u: {sample['u'].shape}")
    print(f"  a_ij: {sample['a_ij'].shape}")
    
    # Create coordinate grid
    N = metadata['N']
    coords = create_coordinate_grid(N)
    print(f"\nCoordinate grid shape: {coords.shape}")
    
    # Test DataLoader
    dataloader = get_dataloader('data', K_value=K, batch_size=4)
    batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"  f: {batch['f'].shape}")
    print(f"  u: {batch['u'].shape}")
    
    print("\n✓ All tests passed!")