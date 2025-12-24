"""
Data Generation Script for Multiscale Poisson Equation

This script generates a dataset of input-output pairs (f^(i), u^(i)) for the 2D Poisson equation described in the Problem statement in the README.

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


class PoissonDataGenerator:
    
    def __init__(self, N=64, r=0.5, domain=[0, 1]):
        """
        Initialize the data generator.
        
        Parameters:

        N : int
            Grid size (N x N)
        r : float
            Exponent for frequency scaling (default: 0.5)
        domain : list
            Domain bounds [x_min, x_max] for square domain
        """
        self.N = N
        self.r = r
        self.domain = domain
        
        # Create spatial grid
        x = np.linspace(domain[0], domain[1], N)
        y = np.linspace(domain[0], domain[1], N)
        self.X, self.Y = np.meshgrid(x, y)
        
    def generate_sample(self, K, seed=None):
        """
        Generate a single sample with K frequency modes.
        
        Parameters:

        K : int
            Number of frequency modes (complexity parameter)
        seed : int, optional
            Random seed for reproducibility
            
        Returns:

        f : np.ndarray
            Source field (N x N)
        u : np.ndarray
            Solution field (N x N)
        a_ij : np.ndarray
            Coefficient matrix (K x K)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random coefficients from N(0, 1)
        a_ij = np.random.randn(K, K)
        
        # Initialize fields
        f = np.zeros_like(self.X)
        u = np.zeros_like(self.X)
        
        # Compute source and solution using analytical formulas
        for i in range(1, K + 1):
            for j in range(1, K + 1):
                freq_factor = (i**2 + j**2)
                
                # Sine basis functions
                sin_ix = np.sin(np.pi * i * self.X)
                sin_jy = np.sin(np.pi * j * self.Y)
                basis = sin_ix * sin_jy
                
                # Source term: f(x,y)
                f += a_ij[i-1, j-1] * (freq_factor**self.r) * basis
                
                # Solution: u(x,y)
                u += a_ij[i-1, j-1] * (freq_factor**(self.r - 1)) * basis
        
        # Apply scaling factors
        f *= (np.pi / K**2)
        u *= (1 / (np.pi * K**2))
        
        return f, u, a_ij
    
    def generate_dataset(self, K_values, num_samples_per_K, base_seed=42):
        """
        Generate a complete dataset for multiple K values.
        
        Parameters:
        K_values : list
            List of K values (e.g., [1, 4, 8, 16])
        num_samples_per_K : int
            Number of samples to generate for each K
        base_seed : int
            Base random seed
            
        Returns:
        dataset : dict
            Dictionary containing all generated data
        """
        dataset = {
            'N': self.N,
            'r': self.r,
            'domain': self.domain,
            'K_values': K_values,
            'samples': {}
        }
        
        for K in K_values:
            dataset['samples'][K] = []
            
            for sample_idx in range(num_samples_per_K):
                seed = base_seed + K * 1000 + sample_idx
                f, u, a_ij = self.generate_sample(K, seed=seed)
                
                dataset['samples'][K].append({
                    'f': f,
                    'u': u,
                    'a_ij': a_ij,
                    'seed': seed
                })
                
            print(f"Generated {num_samples_per_K} samples for K={K}")
        
        return dataset
    
    def save_dataset(self, dataset, save_dir='data'):
        """
        Save dataset to disk.
        
        Parameters:
        dataset : dict
            Dataset dictionary
        save_dir : str
            Directory to save data
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Save metadata
        metadata = {
            'N': dataset['N'],
            'r': dataset['r'],
            'domain': dataset['domain'],
            'K_values': dataset['K_values'],
            'num_samples_per_K': len(dataset['samples'][dataset['K_values'][0]])
        }
        
        with open(save_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save samples for each K
        for K in dataset['K_values']:
            K_dir = save_path / f'K_{K}'
            K_dir.mkdir(exist_ok=True)
            
            for idx, sample in enumerate(dataset['samples'][K]):
                np.savez(
                    K_dir / f'sample_{idx}.npz',
                    f=sample['f'],
                    u=sample['u'],
                    a_ij=sample['a_ij'],
                    seed=sample['seed']
                )
        
        print(f"\nDataset saved to {save_path.absolute()}")


def plot_samples(dataset, K_values, num_examples=3, save_path='figures'):
    """
    Plot examples of source and solution fields for different K values.
    
    Parameters:
    dataset : dict
        Generated dataset
    K_values : list
        List of K values to plot
    num_examples : int
        Number of examples to plot for each K
    save_path : str
        Directory to save figures
    """
    save_dir = Path(save_path)
    save_dir.mkdir(exist_ok=True)
    
    for K in K_values:
        fig, axes = plt.subplots(num_examples, 2, figsize=(12, 4*num_examples))
        
        if num_examples == 1:
            axes = axes.reshape(1, -1)
        
        for idx in range(num_examples):
            sample = dataset['samples'][K][idx]
            f = sample['f']
            u = sample['u']
            
            # Plot source field f
            im1 = axes[idx, 0].imshow(f, cmap='RdBu_r', origin='lower', 
                                       extent=[0, 1, 0, 1], aspect='auto')
            axes[idx, 0].set_title(f'Source f (K={K}, Sample {idx+1})', fontsize=12)
            axes[idx, 0].set_xlabel('x')
            axes[idx, 0].set_ylabel('y')
            plt.colorbar(im1, ax=axes[idx, 0])
            
            # Plot solution field u
            im2 = axes[idx, 1].imshow(u, cmap='RdBu_r', origin='lower',
                                       extent=[0, 1, 0, 1], aspect='auto')
            axes[idx, 1].set_title(f'Ground Truth u (K={K}, Sample {idx+1})', fontsize=12)
            axes[idx, 1].set_xlabel('x')
            axes[idx, 1].set_ylabel('y')
            plt.colorbar(im2, ax=axes[idx, 1])
        
        plt.tight_layout()
        plt.savefig(save_dir / f'examples_K_{K}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved figure for K={K}")


def main():
    
    # Configuration
    N = 64  # Grid size
    K_values = [1, 4, 8, 16]  # Frequency modes
    num_samples_per_K = 100  # Number of samples per K value
    num_plot_examples = 3  # Number of examples to plot
    
    print("="*60)
    print("Multiscale Poisson Equation - Data Generation")
    print("="*60)
    print(f"Grid size: {N} x {N}")
    print(f"K values: {K_values}")
    print(f"Samples per K: {num_samples_per_K}")
    print("="*60)
    print()
    
    # Initialize generator
    generator = PoissonDataGenerator(N=N, r=0.5)
    
    # Generate dataset
    print("Generating dataset...")
    dataset = generator.generate_dataset(K_values, num_samples_per_K)
    
    # Save dataset
    print("\nSaving dataset...")
    generator.save_dataset(dataset, save_dir='data')
    
    # Plot examples
    print(f"\nPlotting {num_plot_examples} examples for each K...")
    plot_samples(dataset, K_values, num_examples=num_plot_examples, save_path='figures')
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    for K in K_values:
        samples = dataset['samples'][K]
        f_vals = [s['f'] for s in samples]
        u_vals = [s['u'] for s in samples]
        
        f_mean = np.mean([np.abs(f).max() for f in f_vals])
        u_mean = np.mean([np.abs(u).max() for u in u_vals])
        
        print(f"K={K:2d}: Mean max |f| = {f_mean:.4f}, Mean max |u| = {u_mean:.6f}")
    
    print("\n" + "="*60)
    print("Data generation complete!")
    print("="*60)


if __name__ == "__main__":
    main()