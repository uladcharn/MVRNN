import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

class LatentNormalityTester:
    """
    Test normality of latent representations in mVRNN-WAE.
    """
    def __init__(self, model, z_enc, z_prior, device='cpu'):
        self.model = model
        self.z_enc = z_enc
        self.z_prior = z_prior
        self.n_samples, self.d = z_enc.shape
        self.device = device
        
    def collect_latent_samples(self, data_x, max_samples=5000):
        """
        Collect latent samples from numpy array.
        
        Args:
            data_x: Numpy array [seq_len, input_size] or [seq_len, batch_size, input_size]
            max_samples: Maximum number of samples to collect
            
        Returns:
            z_enc: [n_samples, latent_dim] from encoder
            z_prior: [n_samples, latent_dim] from prior
        """
        
        # Convert to tensor
        data_tensor = torch.from_numpy(data_x).float().to(self.device)
        
        # Check dimensions and reshape if needed
        if data_tensor.dim() == 2:
            # [seq_len, input_size] -> [seq_len, 1, input_size]
            data_tensor = data_tensor.unsqueeze(1)
        elif data_tensor.dim() == 3:
            # Already [seq_len, batch_size, input_size]
            pass
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {data_tensor.shape}")
        
        print(f"Data shape: {data_tensor.shape}")
        
        # Forward pass
        with torch.no_grad():
            _, _, info = self.model(data_tensor, hidden_state=None, sample=True)
        
        # Extract latent samples
        if 'all_z_enc' not in info or 'all_z_prior' not in info:
            raise ValueError("Model info dict does not contain 'all_z_enc' or 'all_z_prior'")
        
        z_enc = info['all_z_enc']      # [seq, batch, latent_dim]
        z_prior = info['all_z_prior']  # [seq, batch, latent_dim]
        
        # Flatten
        z_enc_flat = z_enc.reshape(-1, z_enc.size(-1)).cpu()
        z_prior_flat = z_prior.reshape(-1, z_prior.size(-1)).cpu()
        
        # Limit samples if needed
        if z_enc_flat.size(0) > max_samples:
            indices = torch.randperm(z_enc_flat.size(0))[:max_samples]
            z_enc_flat = z_enc_flat[indices]
            z_prior_flat = z_prior_flat[indices]
        
        print(f"Collected {z_enc_flat.size(0)} latent samples")
        print(f"Latent dimension: {z_enc_flat.size(1)}")
        
        return z_enc_flat.numpy(), z_prior_flat.numpy()
    
    def shapiro_wilk_test(self, samples, alpha=0.05):
        """
        Perform Shapiro-Wilk test for normality.
        
        Args:
            samples: [n_samples, n_dimensions] array
            alpha: Significance level (default 0.05)
            
        Returns:
            results: Dictionary with test results for each dimension
        """
        n_samples, n_dims = samples.shape
        
        # Initialize results dictionary
        results = {
            'dimension': [],      # Dimension index (0, 1, 2, ...)
            'statistic': [],      # Shapiro-Wilk W statistic
            'p_value': [],        # p-value from test
            'is_normal': [],      # True if p > alpha
            'mean': [],           # Empirical mean of dimension
            'std': [],            # Empirical std of dimension
            'skewness': [],       # Skewness (0 for normal)
            'kurtosis': []        # Excess kurtosis (0 for normal)
        }
        
        print(f"\nShapiro-Wilk Test for Normality (α = {alpha})")
        print("=" * 100)
        print(f"{'Dim':<6} {'W-Stat':<10} {'p-value':<12} {'Normal?':<10} "
              f"{'Mean':<10} {'Std':<10} {'Skew':<10} {'Kurt':<10}")
        print("-" * 100)
        
        for dim in range(n_dims):
            dim_samples = samples[:, dim]
            
            # Shapiro-Wilk test
            statistic, p_value = stats.shapiro(dim_samples)
            is_normal = p_value > alpha
            
            # Compute statistics
            mean = np.mean(dim_samples)
            std = np.std(dim_samples)
            skewness = stats.skew(dim_samples)
            kurtosis = stats.kurtosis(dim_samples)  # Excess kurtosis
            
            # Store results
            results['dimension'].append(dim)
            results['statistic'].append(statistic)
            results['p_value'].append(p_value)
            results['is_normal'].append(is_normal)
            results['mean'].append(mean)
            results['std'].append(std)
            results['skewness'].append(skewness)
            results['kurtosis'].append(kurtosis)
            
            # Print row
            normal_str = "✓ Yes" if is_normal else "✗ No"
            print(f"{dim:<6} {statistic:<10.6f} {p_value:<12.6f} {normal_str:<10} "
                  f"{mean:<10.4f} {std:<10.4f} {skewness:<10.4f} {kurtosis:<10.4f}")
        
        print("=" * 100)
        
        # Summary statistics
        n_normal = sum(results['is_normal'])
        normality_rate = n_normal / n_dims
        
        print(f"\nSummary:")
        print(f"  Total dimensions: {n_dims}")
        print(f"  Normal dimensions: {n_normal} ({normality_rate*100:.1f}%)")
        print(f"  Non-normal dimensions: {n_dims - n_normal} ({(1-normality_rate)*100:.1f}%)")
        print(f"  Mean |mean|: {np.mean(np.abs(results['mean'])):.4f} (should be ~0)")
        print(f"  Mean std: {np.mean(results['std']):.4f} (should be ~1)")
        print(f"  Mean |skewness|: {np.mean(np.abs(results['skewness'])):.4f} (should be ~0)")
        print(f"  Mean |kurtosis|: {np.mean(np.abs(results['kurtosis'])):.4f} (should be ~0)")
        
        # Add summary to results
        results['summary'] = {
            'n_dimensions': n_dims,
            'n_normal': n_normal,
            'normality_rate': normality_rate,
            'mean_abs_mean': np.mean(np.abs(results['mean'])),
            'mean_std': np.mean(results['std']),
            'mean_abs_skewness': np.mean(np.abs(results['skewness'])),
            'mean_abs_kurtosis': np.mean(np.abs(results['kurtosis']))
        }
        
        return results
    
    def generate_random_projections(self, n_projections=5000):
        # Sample from standard normal
        projections = np.random.randn(n_projections, self.d)
        
        # Normalize to unit vectors
        norms = np.linalg.norm(projections, axis=1, keepdims=True)
        projections = projections / norms
        
        return projections
    
    def generate_pca_projections(self, n_components=None):
        
        if n_components is None:
            n_components = 10  # Use top 10 or all if d < 10
        
        pca = PCA(n_components=n_components)
        pca.fit(self.z)

        return pca.components_, pca.explained_variance_ratio_
    
    def test_projection_normality(self, projected_data):
        """
        Test normality of 1D projections using Shapiro-Wilk.
        
        Args:
            projected_data: [n_samples, n_proj] array
            
        Returns:
            results: Dict with statistics and p-values
        """
        n_proj = projected_data.shape[1]
        
        results = {
            'p_values': [],
            'statistics': [],
            'means': [],
            'stds': [],
            'skewness': [],
            'kurtosis': []
        }
        
        for i in range(n_proj):
            y = projected_data[:, i]
            
            # Shapiro-Wilk test
            stat, p_val = stats.shapiro(y)
            
            # Summary statistics
            results['p_values'].append(p_val)
            results['statistics'].append(stat)
            results['means'].append(np.mean(y))
            results['stds'].append(np.std(y))
            results['skewness'].append(stats.skew(y))
            results['kurtosis'].append(stats.kurtosis(y))
        
        results['normality_rate'] = np.mean(np.array(results['p_values']) > 0.05)
        
        return results
    
    def plot_p_value_distribution(self, results, component, 
                                  save_path='p_value_dist.png'):
        """
        Plot distribution of p-values across all projections.
        
        Args:
            results: Results from test_projection_normality
            save_path: Where to save figure
        """
        p_values = np.array(results['p_values'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram of p-values
        ax1.hist(p_values, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2, 
                   label='α = 0.05')
        ax1.set_xlabel('p-value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Shapiro-Wilk p-values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ECDF of p-values
        sorted_p = np.sort(p_values)
        ecdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
        ax2.plot(sorted_p, ecdf, linewidth=2, label='Empirical CDF')
        ax2.plot([0, 1], [0, 1], 'k--', label='Uniform (if normal)')
        ax2.axvline(x=0.05, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax2.set_xlabel('p-value')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('ECDF of p-values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add text with normality rate
        normality_rate = results['normality_rate'] * 100
        fig.text(0.5, 0.02, 
                f'{component}: {normality_rate:.1f}% of projections are normal (p > 0.05)',
                ha='center', fontsize=12, weight='bold')
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved p-value distribution to: {save_path}")
        plt.close()

    def plot_latent_distributions(self, z_enc, z_prior, save_path='latent_distributions.png'):
        """
        Visualize latent distributions.
        
        Args:
            z_enc: [n_samples, latent_dim] encoder samples
            z_prior: [n_samples, latent_dim] prior samples
            save_path: Path to save figure
        """
        n_dims = z_enc.shape[1]
        n_cols = min(4, n_dims)
        n_rows = (n_dims + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_dims > 1 else [axes]
        
        for dim in range(n_dims):
            ax = axes[dim]
            
            # Plot histograms
            ax.hist(z_enc[:, dim], bins=50, alpha=0.6, label='Encoder', 
                   density=True, color='blue')
            ax.hist(z_prior[:, dim], bins=50, alpha=0.6, label='Prior', 
                   density=True, color='red')
            
            # Plot theoretical normal
            x = np.linspace(z_enc[:, dim].min(), z_enc[:, dim].max(), 100)
            ax.plot(x, stats.norm.pdf(x, 0, 1), 'k--', label='N(0,1)', linewidth=2)
            
            # Formatting
            ax.set_title(f'Dimension {dim}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_dims, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved distribution plots to: {save_path}")
        plt.close()

    def generate_report(self, enc_results, prior_results, save_path='normality_report.txt'):
        """
        Generate comprehensive text report.
        
        Args:
            enc_results: Results from encoder samples
            prior_results: Results from prior samples
            save_path: Path to save report
        """
        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("LATENT SPACE NORMALITY ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Encoder results
            f.write("ENCODER LATENT CODES (z_enc)\n")
            f.write("-"*80 + "\n")
            f.write(f"Total dimensions: {enc_results['summary']['n_dimensions']}\n")
            f.write(f"Normal dimensions: {enc_results['summary']['n_normal']} "
                   f"({enc_results['summary']['normality_rate']*100:.1f}%)\n")
            f.write(f"Mean |mean|: {enc_results['summary']['mean_abs_mean']:.4f}\n")
            f.write(f"Mean std: {enc_results['summary']['mean_std']:.4f}\n")
            f.write(f"Mean |skewness|: {enc_results['summary']['mean_abs_skewness']:.4f}\n")
            f.write(f"Mean |kurtosis|: {enc_results['summary']['mean_abs_kurtosis']:.4f}\n\n")
            
            # Prior results
            f.write("PRIOR SAMPLES (z_prior)\n")
            f.write("-"*80 + "\n")
            f.write(f"Total dimensions: {prior_results['summary']['n_dimensions']}\n")
            f.write(f"Normal dimensions: {prior_results['summary']['n_normal']} "
                   f"({prior_results['summary']['normality_rate']*100:.1f}%)\n")
            f.write(f"Mean |mean|: {prior_results['summary']['mean_abs_mean']:.4f}\n")
            f.write(f"Mean std: {prior_results['summary']['mean_std']:.4f}\n\n")
            f.write(f"Mean |skewness|: {prior_results['summary']['mean_abs_skewness']:.4f}\n")
            f.write(f"Mean |kurtosis|: {prior_results['summary']['mean_abs_kurtosis']:.4f}\n\n")
            
            # Interpretation
            f.write("INTERPRETATION\n")
            f.write("-"*80 + "\n")
            
            enc_rate = enc_results['summary']['normality_rate']
            if enc_rate > 0.9:
                f.write("✓ EXCELLENT: >90% of latent dimensions are normally distributed.\n")
                f.write("  The WAE successfully matched the aggregated posterior to the prior.\n")
            elif enc_rate > 0.75:
                f.write("✓ GOOD: >75% of latent dimensions are normally distributed.\n")
                f.write("  The WAE is working well, minor improvements possible.\n")
            elif enc_rate > 0.5:
                f.write("⚠ MODERATE: 50-75% of latent dimensions are normal.\n")
                f.write("  Consider: increasing lambda_mmd or training longer.\n")
            else:
                f.write("✗ POOR: <50% of latent dimensions are normal.\n")
                f.write("  The MMD regularization may be too weak or training insufficient.\n")
                f.write("  Recommendations:\n")
                f.write("  - Increase lambda_mmd (current value may be too small)\n")
                f.write("  - Adjust kernel bandwidth sigma\n")
                f.write("  - Train for more epochs\n")
                f.write("  - Check for posterior collapse\n")
            
            f.write("\n")
            
            # Detailed per-dimension results
            f.write("DETAILED RESULTS PER DIMENSION - ENCODER\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Dim':<5} {'W-Stat':<10} {'p-value':<12} {'Normal?':<10} "
                   f"{'Mean':<10} {'Std':<10}\n")
            f.write("-"*80 + "\n")
            
            for i in range(len(enc_results['dimension'])):
                normal_str = "Yes" if enc_results['is_normal'][i] else "No"
                f.write(f"{enc_results['dimension'][i]:<5} "
                       f"{enc_results['statistic'][i]:<10.6f} "
                       f"{enc_results['p_value'][i]:<12.6f} "
                       f"{normal_str:<10} "
                       f"{enc_results['mean'][i]:<10.4f} "
                       f"{enc_results['std'][i]:<10.4f}\n")
                
            f.write("DETAILED RESULTS PER DIMENSION - PRIOR\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Dim':<5} {'W-Stat':<10} {'p-value':<12} {'Normal?':<10} "
                   f"{'Mean':<10} {'Std':<10}\n")
            f.write("-"*80 + "\n")
                
            for i in range(len(prior_results['dimension'])):
                normal_str = "Yes" if prior_results['is_normal'][i] else "No"
                f.write(f"{prior_results['dimension'][i]:<5} "
                       f"{prior_results['statistic'][i]:<10.6f} "
                       f"{prior_results['p_value'][i]:<12.6f} "
                       f"{normal_str:<10} "
                       f"{prior_results['mean'][i]:<10.4f} "
                       f"{prior_results['std'][i]:<10.4f}\n")
        
        print(f"\nSaved report to: {save_path}")
    
    def plot_qq_plots(self, z_enc, save_path='qq_plots.png'):
        """
        Create Q-Q plots for each latent dimension.
        
        Args:
            z_enc: [n_samples, latent_dim] encoder samples
            save_path: Path to save figure
        """
        n_dims = z_enc.shape[1]
        n_cols = min(4, n_dims)
        n_rows = (n_dims + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_dims > 1 else [axes]
        
        for dim in range(n_dims):
            ax = axes[dim]
            
            # Q-Q plot
            stats.probplot(z_enc[:, dim], dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot: Dimension {dim}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_dims, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Q-Q plots to: {save_path}")
        plt.close()