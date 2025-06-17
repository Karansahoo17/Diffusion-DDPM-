import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
from typing import Dict, Any, Optional
import wandb

from ..model.diffusion import DDPM
from ..utils.helpers import save_checkpoint, load_checkpoint
from ..utils.visualization import save_samples
from .losses import DDPMLoss


class DDPMTrainer:
    """Trainer class for DDPM model."""
    
    def __init__(
        self,
        model: DDPM,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Dict[str, Any] = None,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.device = device
        
        # Training components
        self.criterion = DDPMLoss()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.logger = self._setup_logger()
        self.use_wandb = self.config.get('use_wandb', False)
        
        if self.use_wandb:
            wandb.init(
                project=self.config.get('project_name', 'ddpm'),
                config=self.config
            )
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        lr = optimizer_config.get('lr', 1e-4)
        weight_decay = optimizer_config.get('weight_decay', 0.0)
        
        if optimizer_type.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        if not scheduler_config.get('use_scheduler', False):
            return None
            
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            return None
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger."""
        logger = logging.getLogger('DDPMTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss = self.model.training_step(x)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
            
            # Log to wandb
            if self.use_wandb and self.global_step % self.config.get('log_freq', 100) == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step
                })
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                
                loss = self.model.training_step(x)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        save_checkpoint(checkpoint, filepath, is_best)
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = load_checkpoint(filepath, self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded: {filepath}")
    
    def generate_samples(self, num_samples: int = 8, save_path: str = None):
        """Generate and save samples."""
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(
                num_samples=num_samples,
                device=self.device
            )
        
        if save_path:
            save_samples(samples, save_path)
        
        return samples
    
    def train(self, epochs: int, save_dir: str = './checkpoints'):
        """Main training loop."""
        self.logger.info(f"Starting training for {epochs} epochs")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            self.logger.info(
                f"Epoch {epoch}: " + 
                " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            )
            
            if self.use_wandb:
                wandb.log({**metrics, 'epoch': epoch})
            
            # Save checkpoints
            is_best = False
            if val_metrics and val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                is_best = True
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_freq', 10) == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                self.save_checkpoint(checkpoint_path, is_best)
            
            # Generate samples
            if (epoch + 1) % self.config.get('sample_freq', 20) == 0:
                sample_path = os.path.join(save_dir, f'samples_epoch_{epoch+1}.png')
                self.generate_samples(save_path=sample_path)
        
        self.logger.info("Training completed!")
        
        # Save final checkpoint
        final_checkpoint_path = os.path.join(save_dir, 'final_checkpoint.pth')
        self.save_checkpoint(final_checkpoint_path, False)
        
        if self.use_wandb:
            wandb.finish()


def create_trainer(model, train_loader, val_loader=None, config=None, device='cuda'):
    """Factory function to create trainer."""
    return DDPMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
