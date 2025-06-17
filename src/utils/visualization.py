"""
Visualization utilities for DDPM training and sampling.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from typing import List, Optional, Tuple
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8')


class DDPMVisualizer:
    """Handles visualization for DDPM training and sampling."""
    
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def plot_training_curves(self, 
                           train_losses: List[float],
                           val_losses: Optional[List[float]] = None,
                           save_path: Optional[str] = None) -> None:
        """Plot training and validation loss curves."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        
        if val_losses is not None:
            ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('DDPM Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_noise_schedule(self, 
                          betas: torch.Tensor,
                          alphas: torch.Tensor,
                          alpha_bars: torch.Tensor,
                          save_path: Optional[str] = None) -> None:
        """Plot the noise schedule parameters."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        timesteps = range(len(betas))
        
        # Beta schedule
        axes[0, 0].plot(timesteps, betas.cpu().numpy(), 'b-', linewidth=2)
        axes[0, 0].set_title('Beta Schedule')
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('β_t')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Alpha schedule
        axes[0, 1].plot(timesteps, alphas.cpu().numpy(), 'g-', linewidth=2)
        axes[0, 1].set_title('Alpha Schedule')
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].set_ylabel('α_t')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Alpha bar schedule
        axes[1, 0].plot(timesteps, alpha_bars.cpu().numpy(), 'r-', linewidth=2)
        axes[1, 0].set_title('Cumulative Alpha Schedule')
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('ᾱ_t')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Signal-to-noise ratio
        snr = alpha_bars / (1 - alpha_bars)
        axes[1, 1].plot(timesteps, snr.cpu().numpy(), 'm-', linewidth=2)
        axes[1, 1].set_title('Signal-to-Noise Ratio')
        axes[1, 1].set_xlabel('Timestep')
        axes[1, 1].set_ylabel('SNR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_dir / 'noise_schedule.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_diffusion_process(self, 
                             original_images: torch.Tensor,
                             noisy_images: List[torch.Tensor],
                             timesteps: List[int],
                             save_path: Optional[str] = None) -> None:
        """Visualize the forward diffusion process."""
        n_samples = min(8, original_images.shape[0])
        n_timesteps = len(timesteps)
        
        fig, axes = plt.subplots(n_samples, n_timesteps + 1, 
                               figsize=(2 * (n_timesteps + 1), 2 * n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            # Original image
            img = self._tensor_to_image(original_images[i])
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'Original' if i == 0 else '')
            axes[i, 0].axis('off')
            
            # Noisy images at different timesteps
            for j, (noisy_img, t) in enumerate(zip(noisy_images, timesteps)):
                img = self._tensor_to_image(noisy_img[i])
                axes[i, j + 1].imshow(img)
                axes[i, j + 1].set_title(f't={t}' if i == 0 else '')
                axes[i, j + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_dir / 'diffusion_process.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sampling_process(self, 
                            sampling_steps: List[torch.Tensor],
                            timesteps: List[int],
                            save_path: Optional[str] = None) -> None:
        """Visualize the reverse sampling process."""
        n_samples = min(8, sampling_steps[0].shape[0])
        n_steps = len(sampling_steps)
        
        # Show every nth step to avoid too many columns
        step_interval = max(1, n_steps // 10)
        selected_steps = sampling_steps[::step_interval]
        selected_timesteps = timesteps[::step_interval]
        
        fig, axes = plt.subplots(n_samples, len(selected_steps), 
                               figsize=(2 * len(selected_steps), 2 * n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            for j, (step_img, t) in enumerate(zip(selected_steps, selected_timesteps)):
                img = self._tensor_to_image(step_img[i])
                axes[i, j].imshow(img)
                axes[i, j].set_title(f't={t}' if i == 0 else '')
                axes[i, j].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_dir / 'sampling_process.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_generated_samples(self, 
                             samples: torch.Tensor,
                             n_rows: int = 8,
                             save_path: Optional[str] = None) -> None:
        """Plot a grid of generated samples."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Create grid
        grid = vutils.make_grid(samples, nrow=n_rows, normalize=True, padding=2)
        
        # Convert to numpy and transpose for matplotlib
        grid_np = grid.cpu().numpy()
        if grid_np.shape[0] == 3:  # RGB
            grid_np = np.transpose(grid_np, (1, 2, 0))
        elif grid_np.shape[0] == 1:  # Grayscale
            grid_np = grid_np[0]
        
        ax.imshow(grid_np, cmap='gray' if len(grid_np.shape) == 2 else None)
        ax.set_title('Generated Samples')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_dir / 'generated_samples.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_loss_components(self, 
                           losses: dict,
                           save_path: Optional[str] = None) -> None:
        """Plot different loss components over time."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        for loss_name, loss_values in losses.items():
            ax.plot(loss_values, label=loss_name, linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_dir / 'loss_components.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_timestep_loss_distribution(self, 
                                      timestep_losses: dict,
                                      save_path: Optional[str] = None) -> None:
        """Plot loss distribution across timesteps."""
        timesteps = list(timestep_losses.keys())
        losses = list(timestep_losses.values())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Line plot
        ax1.plot(timesteps, losses, 'b-', linewidth=2)
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Average Loss')
        ax1.set_title('Loss vs Timestep')
        ax1.grid(True, alpha=0.3)
        
        # Histogram
        ax2.hist(losses, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Loss Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Loss Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_dir / 'timestep_loss_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to displayable image."""
        # Move to CPU and convert to numpy
        img = tensor.cpu().detach().numpy()
        
        # Handle different tensor formats
        if len(img.shape) == 3:
            if img.shape[0] == 1:  # Grayscale
                img = img[0]
            elif img.shape[0] == 3:  # RGB
                img = np.transpose(img, (1, 2, 0))
        
        # Normalize to [0, 1]
        img = (img + 1) / 2  # Assuming tensor is in [-1, 1]
        img = np.clip(img, 0, 1)
        
        return img
    
    def save_checkpoint_visualization(self, 
                                    epoch: int,
                                    train_loss: float,
                                    samples: torch.Tensor) -> None:
        """Save visualization at checkpoint."""
        checkpoint_dir = self.save_dir / f'checkpoint_epoch_{epoch}'
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save sample grid
        self.plot_generated_samples(
            samples, 
            save_path=checkpoint_dir / f'samples_epoch_{epoch}.png'
        )
        
        # Save loss info
        with open(checkpoint_dir / f'loss_epoch_{epoch}.txt', 'w') as f:
            f.write(f'Epoch: {epoch}\n')
            f.write(f'Training Loss: {train_loss:.6f}\n')


def create_comparison_plot(real_images: torch.Tensor, 
                         generated_images: torch.Tensor,
                         save_path: Optional[str] = None) -> None:
    """Create side-by-side comparison of real and generated images."""
    n_samples = min(8, real_images.shape[0], generated_images.shape[0])
    
    fig, axes = plt.subplots(2, n_samples, figsize=(2 * n_samples, 4))
    
    for i in range(n_samples):
        # Real images
        real_img = DDPMVisualizer()._tensor_to_image(real_images[i])
        axes[0, i].imshow(real_img, cmap='gray' if len(real_img.shape) == 2 else None)
        axes[0, i].set_title('Real' if i == 0 else '')
        axes[0, i].axis('off')
        
        # Generated images
        gen_img = DDPMVisualizer()._tensor_to_image(generated_images[i])
        axes[1, i].imshow(gen_img, cmap='gray' if len(gen_img.shape) == 2 else None)
        axes[1, i].set_title('Generated' if i == 0 else '')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig('real_vs_generated.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_architecture_summary(model, input_shape: Tuple[int, ...]) -> None:
    """Plot a summary of model architecture (simplified version)."""
    try:
        from torchsummary import summary
        summary(model, input_shape)
    except ImportError:
        print("torchsummary not available. Install with: pip install torchsummary")
        
        # Fallback: count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
