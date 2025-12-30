"""
Loss Landscape Visualization

This script generates loss landscape visualizations for all trained models
(PINN and Data-Driven, K=1, 4, 16) and creates comparison plots.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from visualize_loss_landscape import visualize_single_landscape
from model_checkpoint import find_checkpoints


def create_comparison_grid(results_list, save_path):
    """
    Create a grid comparing all landscapes side-by-side.
    
    Parameters:
    results_list : list of dict
        List of results from visualize_single_landscape
    save_path : Path or str
        Where to save the comparison plot
    """
    # Organize results by method and K
    results_dict = {}
    for res in results_list:
        method = res['method']
        K = res['K']
        if method not in results_dict:
            results_dict[method] = {}
        results_dict[method][K] = res
    
    # Create figure with 2 rows (methods) × 3 cols (K values)
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    K_values = sorted([1, 4, 16])
    methods = ['data_driven', 'pinn']
    method_names = ['Data-Driven', 'PINN']
    
    for i, (method, method_name) in enumerate(zip(methods, method_names)):
        for j, K in enumerate(K_values):
            ax = fig.add_subplot(gs[i, j])
            
            if method in results_dict and K in results_dict[method]:
                res = results_dict[method][K]
                
                # Plot contour
                Alpha, Beta = np.meshgrid(res['alpha_range'], res['beta_range'])
                loss_grid = res['loss_grid']
                
                # Cap extreme values
                loss_plot = np.clip(loss_grid, loss_grid.min(), 
                                   np.percentile(loss_grid, 95))
                
                # Contour plot
                levels = 15
                contourf = ax.contourf(Alpha, Beta, loss_plot, levels=levels,
                                      cmap='RdYlBu_r', alpha=0.8)
                ax.contour(Alpha, Beta, loss_plot, levels=levels,
                          colors='black', alpha=0.3, linewidths=0.5)
                
                # Mark center
                ax.plot(0, 0, 'r*', markersize=12, zorder=10)
                
                # Labels
                ax.set_xlabel('α', fontsize=10)
                if j == 0:
                    ax.set_ylabel(f'{method_name}\nβ', fontsize=10)
                else:
                    ax.set_ylabel('β', fontsize=10)
                
                ax.set_title(f'K={K}, Test Error: {res["test_error"]:.6f}',
                           fontsize=11, fontweight='bold')
                
                # Colorbar
                cbar = plt.colorbar(contourf, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Loss', fontsize=9)
            else:
                ax.text(0.5, 0.5, f'No data for\n{method_name}, K={K}',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=11)
                ax.set_xlabel('α', fontsize=10)
                ax.set_ylabel('β', fontsize=10)
    
    fig.suptitle('Loss Landscape Comparison: PINN vs Data-Driven\n'
                'Visualizing Spectral Bias and Optimization Complexity',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Comparison grid saved: {save_path}")
    plt.close()


def analyze_landscapes(results_list, save_dir):
    """
    Analyze and compare landscape properties.
    
    Parameters:
    results_list : list of dict
        List of results from visualize_single_landscape
    save_dir : Path
        Directory to save analysis
    """
    save_dir = Path(save_dir)
    
    print("\n" + "="*70)
    print("LANDSCAPE ANALYSIS")
    print("="*70)
    
    # Organize by method and K
    analysis = {}
    for res in results_list:
        method = res['method']
        K = res['K']
        
        loss_grid = res['loss_grid']
        center_idx = (len(res['beta_range']) // 2, len(res['alpha_range']) // 2)
        center_loss = loss_grid[center_idx]
        
        # Compute sharpness metrics
        # 1. Average loss in neighborhood of θ*
        n_alpha = len(res['alpha_range'])
        n_beta = len(res['beta_range'])
        radius = 5  # 5 grid points around center
        
        i_min = max(0, center_idx[0] - radius)
        i_max = min(n_beta, center_idx[0] + radius + 1)
        j_min = max(0, center_idx[1] - radius)
        j_max = min(n_alpha, center_idx[1] + radius + 1)
        
        neighborhood = loss_grid[i_min:i_max, j_min:j_max]
        avg_neighborhood_loss = neighborhood.mean()
        
        # 2. Maximum loss in neighborhood (indicator of sharpness)
        max_neighborhood_loss = neighborhood.max()
        
        # 3. "Flatness" = how much loss increases in neighborhood
        flatness = max_neighborhood_loss - center_loss
        
        # 4. Count "local minima" (crude estimate)
        # Look for points that are lower than all 8 neighbors
        local_minima_count = 0
        for i in range(1, n_beta - 1):
            for j in range(1, n_alpha - 1):
                val = loss_grid[i, j]
                neighbors = [
                    loss_grid[i-1, j-1], loss_grid[i-1, j], loss_grid[i-1, j+1],
                    loss_grid[i, j-1], loss_grid[i, j+1],
                    loss_grid[i+1, j-1], loss_grid[i+1, j], loss_grid[i+1, j+1]
                ]
                if all(val < n for n in neighbors):
                    local_minima_count += 1
        
        # Store analysis
        key = f"{method}_K{K}"
        analysis[key] = {
            'method': method,
            'K': K,
            'test_error': res['test_error'],
            'center_loss': center_loss,
            'avg_neighborhood_loss': avg_neighborhood_loss,
            'max_neighborhood_loss': max_neighborhood_loss,
            'flatness': flatness,
            'local_minima_count': local_minima_count,
            'min_loss': res['min_loss'],
            'max_loss': res['max_loss']
        }
    
    # Print analysis
    print("\n" + "-"*70)
    print(f"{'Method':<15} {'K':<5} {'Test Err':<12} {'Flatness':<12} {'Local Minima'}")
    print("-"*70)
    
    for key in sorted(analysis.keys()):
        a = analysis[key]
        method_name = "Data-Driven" if a['method'] == 'data_driven' else "PINN"
        print(f"{method_name:<15} {a['K']:<5} {a['test_error']:<12.6f} "
              f"{a['flatness']:<12.6f} {a['local_minima_count']}")
    
    print("-"*70)
    
    # Observations
    print("\nKEY OBSERVATIONS:")
    print("-"*70)
    
    # Compare PINN vs Data-Driven for each K
    for K in [1, 4, 16]:
        pinn_key = f"pinn_K{K}"
        data_key = f"data_driven_K{K}"
        
        if pinn_key in analysis and data_key in analysis:
            pinn = analysis[pinn_key]
            data = analysis[data_key]
            
            print(f"\nK={K}:")
            if pinn['flatness'] > data['flatness']:
                print(f"  - PINN landscape is SHARPER (flatness: {pinn['flatness']:.4f} vs {data['flatness']:.4f})")
            else:
                print(f"  - Data-Driven landscape is SHARPER (flatness: {data['flatness']:.4f} vs {pinn['flatness']:.4f})")
            
            if pinn['local_minima_count'] > data['local_minima_count']:
                print(f"  - PINN has MORE local minima ({pinn['local_minima_count']} vs {data['local_minima_count']})")
            else:
                print(f"  - Data-Driven has MORE local minima ({data['local_minima_count']} vs {pinn['local_minima_count']})")
            
            if pinn['test_error'] > data['test_error']:
                print(f"  - PINN generalizes WORSE (error: {pinn['test_error']:.6f} vs {data['test_error']:.6f})")
            else:
                print(f"  - Data-Driven generalizes WORSE (error: {data['test_error']:.6f} vs {pinn['test_error']:.6f})")
    
    # Compare across K values for each method
    for method in ['pinn', 'data_driven']:
        method_name = "PINN" if method == 'pinn' else "Data-Driven"
        print(f"\n{method_name} across K values:")
        
        K_flatness = {}
        K_minima = {}
        for K in [1, 4, 16]:
            key = f"{method}_K{K}"
            if key in analysis:
                K_flatness[K] = analysis[key]['flatness']
                K_minima[K] = analysis[key]['local_minima_count']
        
        if len(K_flatness) >= 2:
            if K_flatness[16] > K_flatness[1]:
                print(f"  - Landscape becomes SHARPER as K increases")
            else:
                print(f"  - Landscape becomes FLATTER as K increases")
            
            if K_minima[16] > K_minima[1]:
                print(f"  - More local minima as K increases (spectral bias!)")
            else:
                print(f"  - Fewer local minima as K increases")
    
    print("="*70)
    
    # Save analysis to JSON
    import json
    analysis_path = save_dir / 'landscape_analysis.json'
    with open(analysis_path, 'w') as f:
        # Convert numpy types to Python types
        analysis_save = {}
        for key, val in analysis.items():
            analysis_save[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                 for k, v in val.items()}
        json.dump(analysis_save, f, indent=2)
    
    print(f"\n✓ Analysis saved: {analysis_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Loss Landscapes for PINN vs Data-Driven'
    )
    parser.add_argument('--results_dir', type=str, default='results_single_instance',
                       help='Directory containing trained model checkpoints')
    parser.add_argument('--instance_dir', type=str, default='single_instance',
                       help='Directory containing instance data')
    parser.add_argument('--K_values', nargs='+', type=int, default=[1, 4, 16],
                       help='K values to visualize')
    parser.add_argument('--seed', type=int, default=11,
                       help='Seed used for instances')
    parser.add_argument('--resolution', type=int, default=51,
                       help='Grid resolution (51 is standard, 101 is high-res)')
    parser.add_argument('--alpha_range', type=float, nargs=2, default=[-1, 1],
                       help='Range for alpha direction')
    parser.add_argument('--beta_range', type=float, nargs=2, default=[-1, 1],
                       help='Range for beta direction')
    parser.add_argument('--save_dir', type=str, default='visualization_results',
                       help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LOSS LANDSCAPE VISUALIZATION")
    print("="*70)
    print(f"K values: {args.K_values}")
    print(f"Resolution: {args.resolution} × {args.resolution}")
    print(f"α range: {args.alpha_range}")
    print(f"β range: {args.beta_range}")
    print(f"Device: {args.device}")
    print("="*70)
    
    results_list = []
    
    # Visualize each method and K value
    for K in args.K_values:
        for method in ['data_driven', 'pinn']:
            print(f"\n\n{'#'*70}")
            print(f"Processing: {method.upper()}, K={K}")
            print(f"{'#'*70}")
            
            # Find checkpoint
            checkpoint_pattern = f"{method}_K{K}_seed{args.seed}_checkpoint.pt"
            checkpoint_dir = Path(args.results_dir) / f"K{K}_seed{args.seed}"
            checkpoint_path = checkpoint_dir / checkpoint_pattern
            
            # Find instance
            instance_pattern = f"instance_K{K}_seed{args.seed}.npz"
            instance_path = Path(args.instance_dir) / instance_pattern
            
            # Check if files exist
            if not checkpoint_path.exists():
                print(f"⚠ Warning: Checkpoint not found: {checkpoint_path}")
                print(f"  Skipping {method}, K={K}")
                continue
            
            if not instance_path.exists():
                print(f"⚠ Warning: Instance not found: {instance_path}")
                print(f"  Skipping {method}, K={K}")
                continue
            
            # Visualize
            try:
                results = visualize_single_landscape(
                    checkpoint_path=checkpoint_path,
                    instance_path=instance_path,
                    method=method,
                    K=K,
                    alpha_min=args.alpha_range[0],
                    alpha_max=args.alpha_range[1],
                    n_alpha=args.resolution,
                    beta_min=args.beta_range[0],
                    beta_max=args.beta_range[1],
                    n_beta=args.resolution,
                    device=args.device,
                    save_dir=args.save_dir
                )
                
                results_list.append(results)
                
            except Exception as e:
                print(f" Error processing {method}, K={K}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Create comparison visualizations
    if len(results_list) > 0:
        print(f"\n\n{'#'*70}")
        print("CREATING COMPARISON VISUALIZATIONS")
        print(f"{'#'*70}")
        
        save_dir = Path(args.save_dir)
        
        # Comparison grid
        comparison_path = save_dir / 'comparison_all.png'
        create_comparison_grid(results_list, comparison_path)
        
        # Analysis
        analyze_landscapes(results_list, save_dir)
        
        print(f"\n\n{'='*70}")
        print("VISUALIZATION COMPLETE!")
        print(f"{'='*70}")
        print(f"Results saved in: {save_dir}")
        print(f"\nGenerated files:")
        print(f"  - Individual contour plots: {save_dir}/*_contour.png")
        print(f"  - Individual 3D plots: {save_dir}/*_surface.png")
        print(f"  - Comparison grid: {comparison_path}")
        print(f"  - Analysis: {save_dir}/landscape_analysis.json")
        print(f"{'='*70}")
    else:
        print("\n No results generated. Please check that you have:")
        print("  1. Trained models saved in", args.results_dir)
        print("  2. Instance data in", args.instance_dir)


if __name__ == "__main__":
    main()