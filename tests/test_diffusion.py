import pytest
import torch
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.diffusion import DDPM
from model.unet import UNet
from utils.scheduler import NoiseScheduler
from training.losses import DDPMLoss

class TestDiffusionProcess:
    """Test cases for diffusion process components"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timesteps = 100
        self.batch_size = 4
        self.channels = 3
        self.image_size = 32
        
        # Create a simple U-Net for testing
        self.unet = UNet(
            in_channels=self.channels,
            out_channels=self.channels,
            time_embedding_dim=128,
            down_channels=[64, 128],
            up_channels=[128, 64],
            down_sample=[True, False],
            attn_heads=4
        ).to(self.device)
        
    def test_noise_scheduler(self):
        """Test noise scheduler functionality"""
        scheduler = NoiseScheduler(
            timesteps=self.timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            schedule_type='linear'
        )
        
        # Test beta schedule
        betas = scheduler.get_betas()
        assert len(betas) == self.timesteps, "Beta schedule length mismatch"
        assert betas[0] < betas[-1], "Beta schedule should be increasing"
        assert torch.all(betas > 0), "All betas should be positive"
        assert torch.all(betas < 1), "All betas should be less than 1"
        
        # Test alpha schedule
        alphas = scheduler.get_alphas()
        alphas_cumprod = scheduler.get_alphas_cumprod()
        
        assert len(alphas) == self.timesteps, "Alpha schedule length mismatch"
        assert len(alphas_cumprod) == self.timesteps, "Alpha cumprod length mismatch"
        assert torch.all(alphas > 0), "All alphas should be positive"
        assert torch.all(alphas < 1), "All alphas should be less than 1"
        
    def test_cosine_scheduler(self):
        """Test cosine noise scheduler"""
        scheduler = NoiseScheduler(
            timesteps=self.timesteps,
            schedule_type='cosine',
            s=0.008
        )
        
        betas = scheduler.get_betas()
        alphas_cumprod = scheduler.get_alphas_cumprod()
        
        # Cosine schedule should have different properties
        assert len(betas) == self.timesteps
        assert torch.all(betas > 0)
        assert torch.all(betas < 1)
        
        # Check monotonicity of cumulative alphas
        diff = torch.diff(alphas_cumprod)
        assert torch.all(diff <= 0), "Cumulative alphas should be non-increasing"
        
    def test_forward_diffusion_consistency(self):
        """Test forward diffusion process consistency"""
        ddpm = DDPM(
            unet=self.unet,
            timesteps=self.timesteps,
            beta_start=0.0001,
            beta_end=0.02
        ).to(self.device)
        
        x0 = torch.randn(self.batch_size, self.channels, self.image_size, self.image_size).to(self.device)
        
        # Test at different timesteps
        for t_val in [0, self.timesteps//4, self.timesteps//2, self.timesteps-1]:
            t = torch.full((self.batch_size,), t_val, dtype=torch.long).to(self.device)
            noise = torch.randn_like(x0)
            
            xt, _ = ddpm.forward_diffusion(x0, t, noise)
            
            # Check that noise increases with timestep
            if t_val == 0:
                # At t=0, output should be close to input
                assert torch.allclose(xt, x0, atol=1e-3), "At t=0, x_t should equal x_0"
            
            assert xt.shape == x0.shape, f"Shape mismatch at t={t_val}"
            assert torch.isfinite(xt).all(), f"Non-finite values at t={t_val}"
            
    def test_reverse_diffusion_properties(self):
        """Test reverse diffusion process properties"""
        ddpm = DDPM(
            unet=self.unet,
            timesteps=50,  # Smaller for faster testing
            beta_start=0.0001,
            beta_end=0.02
        ).to(self.device)
        
        # Start from pure noise
        xt = torch.randn(1, self.channels, 16, 16).to(self.device)
        
        # Run reverse process for a few steps
        for t in reversed(range(45, 50)):  # Just test last few steps
            t_tensor = torch.full((1,), t, dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                # Predict noise
                predicted_noise = ddpm.unet(xt, t_tensor)
                
                # Compute denoised sample
                alpha = ddpm.alphas[t]
                alpha_cumprod = ddpm.alphas_cumprod[t]
                beta = ddpm.betas[t]
                
                # Denoising step
                if t > 0:
                    noise = torch.randn_like(xt)
                    sigma = torch.sqrt(beta)
                else:
                    noise = torch.zeros_like(xt)
                    sigma = 0
                
                xt = (1 / torch.sqrt(alpha)) * (xt - beta / torch.sqrt(1 - alpha_cumprod) * predicted_noise) + sigma * noise
                
            assert torch.isfinite(xt).all(), f"Non-finite values in reverse step t={t}"
            
    def test_ddpm_loss_function(self):
        """Test DDPM loss function"""
        loss_fn = DDPMLoss()
        
        # Create test data
        predicted_noise = torch.randn(self.batch_size, self.channels, self.image_size, self.image_size)
        target_noise = torch.randn(self.batch_size, self.channels, self.image_size, self.image_size)
        
        # Compute loss
        loss = loss_fn(predicted_noise, target_noise)
        
        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.requires_grad, "Loss should require gradients"
        
        # Test that identical inputs give zero loss
        zero_loss = loss_fn(predicted_noise, predicted_noise)
        assert zero_loss.item() < 1e-6, "Identical inputs should give near-zero loss"
        
    def test_sampling_determinism(self):
        """Test sampling determinism with fixed seed"""
        ddpm = DDPM(
            unet=self.unet,
            timesteps=20,  # Small for faster testing
            beta_start=0.0001,
            beta_end=0.02
        ).to(self.device)
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        sample1 = ddpm.sample(
            batch_size=1,
            channels=3,
            height=16,
            width=16,
            device=self.device
        )
        
        # Set same seed
        torch.manual_seed(42)
        sample2 = ddpm.sample(
            batch_size=1,
            channels=3,
            height=16,
            width=16,
            device=self.device
        )
        
        # Should be identical (or very close due to numerical precision)
        assert torch.allclose(sample1, sample2, atol=1e-5), "Sampling should be deterministic with fixed seed"
        
    def test_progressive_denoising(self):
        """Test that progressive denoising reduces noise"""
        ddpm = DDPM(
            unet=self.unet,
            timesteps=self.timesteps,
            beta_start=0.0001,
            beta_end=0.02
        ).to(self.device)
        
        # Create clean image
        x0 = torch.randn(1, self.channels, 16, 16).to(self.device)
        
        # Add noise at high timestep
        t_high = torch.full((1,), self.timesteps-1, dtype=torch.long).to(self.device)
        noise = torch.randn_like(x0)
        xt_high, _ = ddpm.forward_diffusion(x0, t_high, noise)
        
        # Add noise at low timestep
        t_low = torch.full((1,), self.timesteps//4, dtype=torch.long).to(self.device)
        xt_low, _ = ddpm.forward_diffusion(x0, t_low, noise)
        
        # Higher timestep should be noisier (further from original)
        dist_high = torch.norm(xt_high - x0)
        dist_low = torch.norm(xt_low - x0)
        
        assert dist_high > dist_low, "Higher timestep should be noisier"
        
    def test_model_training_mode(self):
        """Test model behavior in training vs eval mode"""
        ddpm = DDPM(
            unet=self.unet,
            timesteps=self.timesteps
        ).to(self.device)
        
        x0 = torch.randn(2, self.channels, 16, 16).to(self.device)
        
        # Training mode
        ddpm.train()
        loss_train = ddpm.compute_loss(x0)
        
        # Eval mode
        ddpm.eval()
        with torch.no_grad():
            loss_eval = ddpm.compute_loss(x0)
        
        # Both should be finite
        assert torch.isfinite(loss_train), "Training loss should be finite"
        assert torch.isfinite(loss_eval), "Eval loss should be finite"
        
        # Training loss should require gradients
        assert loss_train.requires_grad, "Training loss should require gradients"
        assert not loss_eval.requires_grad, "Eval loss should not require gradients"

class TestDiffusionMath:
    """Test mathematical properties of diffusion"""
    
    def test_variance_schedule_properties(self):
        """Test mathematical properties of variance schedule"""
        scheduler = NoiseScheduler(timesteps=1000, beta_start=0.0001, beta_end=0.02)
        
        betas = scheduler.get_betas()
        alphas = scheduler.get_alphas()
        alphas_cumprod = scheduler.get_alphas_cumprod()
        
        # Test relationship: alpha = 1 - beta
        expected_alphas = 1 - betas
        assert torch.allclose(alphas, expected_alphas, atol=1e-6), "Alpha-beta relationship incorrect"
        
        # Test cumulative product
        expected_cumprod = torch.cumprod(alphas, dim=0)
        assert torch.allclose(alphas_cumprod, expected_cumprod, atol=1e-6), "Cumulative product incorrect"
        
        # Test that cumulative alphas decrease
        assert torch.all(torch.diff(alphas_cumprod) <= 0), "Cumulative alphas should be non-increasing"
        
    def test_signal_to_noise_ratio(self):
        """Test signal-to-noise ratio properties"""
        scheduler = NoiseScheduler(timesteps=1000)
        alphas_cumprod = scheduler.get_alphas_cumprod()
        
        # Signal-to-noise ratio should decrease with time
        snr = alphas_cumprod / (1 - alphas_cumprod)
        
        assert torch.all(torch.diff(snr) <= 0), "SNR should decrease with time"
        assert snr[0] > snr[-1], "SNR should be higher at start than at end"
        
if __name__ == "__main__":
    pytest.main([__file__])
