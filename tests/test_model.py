import pytest
import torch
import torch.nn as nn
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.unet import UNet
from model.diffusion import DDPM

class TestUNet:
    """Test cases for U-Net model"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.channels = 3
        self.image_size = 32
        
    def test_unet_forward_pass(self):
        """Test U-Net forward pass with different configurations"""
        # Test basic configuration
        model = UNet(
            in_channels=3,
            out_channels=3,
            time_embedding_dim=128,
            down_channels=[64, 128, 256],
            up_channels=[256, 128, 64],
            down_sample=[True, True, False],
            attn_heads=4
        ).to(self.device)
        
        # Create test inputs
        x = torch.randn(self.batch_size, self.channels, self.image_size, self.image_size).to(self.device)
        t = torch.randint(0, 1000, (self.batch_size,)).to(self.device)
        
        # Forward pass
        output = model(x, t)
        
        # Check output shape
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains infinite values"
        
    def test_unet_different_sizes(self):
        """Test U-Net with different input sizes"""
        model = UNet(
            in_channels=1,
            out_channels=1,
            time_embedding_dim=64,
            down_channels=[32, 64],
            up_channels=[64, 32],
            down_sample=[True, False],
            attn_heads=2
        ).to(self.device)
        
        # Test different image sizes
        for size in [16, 32, 64]:
            x = torch.randn(1, 1, size, size).to(self.device)
            t = torch.randint(0, 100, (1,)).to(self.device)
            output = model(x, t)
            
            assert output.shape == (1, 1, size, size), f"Failed for size {size}"
    
    def test_unet_gradient_flow(self):
        """Test gradient flow through U-Net"""
        model = UNet(
            in_channels=3,
            out_channels=3,
            time_embedding_dim=128,
            down_channels=[64, 128],
            up_channels=[128, 64],
            down_sample=[True, False],
            attn_heads=4
        ).to(self.device)
        
        x = torch.randn(1, 3, 32, 32, requires_grad=True).to(self.device)
        t = torch.randint(0, 1000, (1,)).to(self.device)
        
        output = model(x, t)
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any(), "Gradient contains NaN"
        
    def test_unet_parameters(self):
        """Test U-Net parameter count and initialization"""
        model = UNet(
            in_channels=3,
            out_channels=3,
            time_embedding_dim=128,
            down_channels=[64, 128, 256],
            up_channels=[256, 128, 64],
            down_sample=[True, True, False],
            attn_heads=4
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0, "Model has no parameters"
        assert trainable_params == total_params, "Some parameters are not trainable"
        
        # Check parameter initialization
        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), f"Parameter {name} contains non-finite values"
            assert not torch.isnan(param).any(), f"Parameter {name} contains NaN values"

class TestDDPM:
    """Test cases for DDPM model"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timesteps = 100
        self.batch_size = 2
        self.channels = 3
        self.image_size = 32
        
    def test_ddpm_initialization(self):
        """Test DDPM initialization"""
        model = DDPM(
            unet=UNet(
                in_channels=3,
                out_channels=3,
                time_embedding_dim=128,
                down_channels=[64, 128],
                up_channels=[128, 64],
                down_sample=[True, False],
                attn_heads=4
            ),
            timesteps=self.timesteps,
            beta_start=0.0001,
            beta_end=0.02
        ).to(self.device)
        
        # Check beta schedule
        assert len(model.betas) == self.timesteps
        assert model.betas[0] < model.betas[-1], "Beta schedule should be increasing"
        assert (model.betas > 0).all(), "All betas should be positive"
        assert (model.betas < 1).all(), "All betas should be less than 1"
        
    def test_ddpm_forward_process(self):
        """Test DDPM forward diffusion process"""
        model = DDPM(
            unet=UNet(
                in_channels=3,
                out_channels=3,
                time_embedding_dim=128,
                down_channels=[64, 128],
                up_channels=[128, 64],
                down_sample=[True, False],
                attn_heads=4
            ),
            timesteps=self.timesteps
        ).to(self.device)
        
        x0 = torch.randn(self.batch_size, self.channels, self.image_size, self.image_size).to(self.device)
        t = torch.randint(0, self.timesteps, (self.batch_size,)).to(self.device)
        
        # Test forward process
        noise = torch.randn_like(x0)
        xt, noise_pred = model.forward_diffusion(x0, t, noise)
        
        assert xt.shape == x0.shape, "Forward diffusion output shape mismatch"
        assert noise_pred.shape == noise.shape, "Noise prediction shape mismatch"
        assert not torch.isnan(xt).any(), "Forward process output contains NaN"
        
    def test_ddpm_sampling(self):
        """Test DDPM sampling process"""
        model = DDPM(
            unet=UNet(
                in_channels=3,
                out_channels=3,
                time_embedding_dim=128,
                down_channels=[32, 64],
                up_channels=[64, 32],
                down_sample=[True, False],
                attn_heads=2
            ),
            timesteps=50  # Smaller for faster testing
        ).to(self.device)
        
        # Test sampling
        samples = model.sample(
            batch_size=1,
            channels=3,
            height=16,
            width=16,
            device=self.device
        )
        
        assert samples.shape == (1, 3, 16, 16), "Sample shape mismatch"
        assert torch.isfinite(samples).all(), "Samples contain non-finite values"
        
    def test_ddpm_loss_computation(self):
        """Test DDPM loss computation"""
        model = DDPM(
            unet=UNet(
                in_channels=3,
                out_channels=3,
                time_embedding_dim=128,
                down_channels=[64, 128],
                up_channels=[128, 64],
                down_sample=[True, False],
                attn_heads=4
            ),
            timesteps=self.timesteps
        ).to(self.device)
        
        x0 = torch.randn(self.batch_size, self.channels, self.image_size, self.image_size).to(self.device)
        
        # Compute loss
        loss = model.compute_loss(x0)
        
        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.requires_grad, "Loss should require gradients"
        
    def test_ddpm_noise_schedule(self):
        """Test DDPM noise schedule properties"""
        model = DDPM(
            unet=UNet(
                in_channels=1,
                out_channels=1,
                time_embedding_dim=64,
                down_channels=[32],
                up_channels=[32],
                down_sample=[False],
                attn_heads=1
            ),
            timesteps=self.timesteps,
            beta_start=0.0001,
            beta_end=0.02
        )
        
        # Test schedule properties
        assert model.alphas.shape == (self.timesteps,)
        assert model.alphas_cumprod.shape == (self.timesteps,)
        assert (model.alphas > 0).all(), "All alphas should be positive"
        assert (model.alphas < 1).all(), "All alphas should be less than 1"
        assert model.alphas_cumprod[-1] < model.alphas_cumprod[0], "Cumulative alphas should decrease"

if __name__ == "__main__":
    pytest.main([__file__])
