#!/usr/bin/env python3
"""
Training script for DDPM model.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from typing import Dict, Any
import os

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model.unet import UNet
from src.model.diffusion import DDPM
from src.data.dataset import get_dataset
from src.training.trainer import DDPMTrainer
from src.utils.scheduler import get_noise_scheduler
from src.utils.helpers import set_seed, setup_logging, save_checkpoint, load_checkpoint
from src.utils.visualization import DDPMVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DDPM model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device to use')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(gpu: int) -> torch.device:
    """Setup device for training."""
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f'cuda:{gpu}')
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def create_model(config: Dict[str, Any], device: torch.device) -> tuple:
    """Create UNet and DDPM models."""
    # Create UNet
    unet = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        features=config['model']['features'],
        n_blocks=config['model']['n_blocks'],
        attention_resolutions=config['model'].get('attention_resolutions', []),
        dropout=config['model'].get('dropout', 0.0),
        use_scale_shift_norm=config['model'].get('use_scale_shift_norm', True)
    ).to(device)
    
    # Create noise scheduler
    scheduler = get_noise_scheduler(
        schedule_type=config['diffusion']['schedule_type'],
        timesteps=config['diffusion']['timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end']
    )
    
    # Create DDPM
    ddpm = DDPM(
        model=unet,
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=scheduler['betas'],
        device=device,
        loss_type=config['training'].get('loss_type', 'mse')
    ).to(device)
    
    return unet, ddpm, scheduler


def create_dataloaders(config: Dict[str, Any]) -> tuple:
    """Create training and validation dataloaders."""
    # Training dataset
    train_dataset = get_dataset(
        dataset_name=config['data']['dataset'],
        data_dir=config['data']['data_dir'],
        image_size=config['data']['image_size'],
        train=True,
        transform_config=config['data'].get('transforms', {})
    )
    
    # Validation dataset
    val_dataset = get_dataset(
        dataset_name=config['data']['dataset'],
        data_dir=config['data']['data_dir'],
        image_size=config['data']['image_size'],
        train=False,
        transform_config=config['data'].get('transforms', {})
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_optimizer_scheduler(model: nn.Module, config: Dict[str, Any]) -> tuple:
    """Create optimizer and learning rate scheduler."""
    # Optimizer
    optimizer_config = config['training']['optimizer']
    if optimizer_config['type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=optimizer_config['lr'],
            betas=optimizer_config.get('betas', (0.9, 0.999)),
            weight_decay=optimizer_config.get('weight_decay', 0)
        )
    elif optimizer_config['type'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=optimizer_config['lr'],
            betas=optimizer_config.get('betas', (0.9, 0.999)),
            weight_decay=optimizer_config.get('weight_decay', 0.01)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_config['type']}")
    
    # Learning rate scheduler
    scheduler = None
    if 'scheduler' in config['training']:
        scheduler_config = config['training']['scheduler']
        if scheduler_config['type'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['training']['epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_config['type'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
    
    return optimizer, scheduler


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup
    set_seed(config.get('seed', 42))
    device = setup_device(args.gpu)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup logging
    setup_logging(output_dir / 'train.log', debug=args.debug)
    logging.info(f"Starting training with config: {args.config}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Device: {device}")
    
    # Initialize Weights & Biases
    if args.wandb:
        wandb.init(
            project=config.get('project_name', 'ddpm'),
            config=config,
            name=config.get('experiment_name', 'ddpm_training')
        )
    
    # Create models
    unet, ddpm, scheduler = create_model(config, device)
    logging.info(f"Created UNet with {sum(p.numel() for p in unet.parameters()):,} parameters")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    logging.info(f"Training samples: {len(train_loader.dataset)}")
    logging.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create optimizer and scheduler
    optimizer, lr_scheduler = create_optimizer_scheduler(unet, config)
    
    # Create visualizer
    visualizer = DDPMVisualizer(save_dir=output_dir / 'visualizations')
    
    # Visualize noise schedule
    visualizer.plot_noise_schedule(
        scheduler['betas'],
        scheduler['alphas'],
        scheduler['alpha_bars'],
        save_path=output_dir / 'noise_schedule.png'
    )
    
    # Create trainer
    trainer = DDPMTrainer(
        model=ddpm,
        optimizer=optimizer,
        device=device,
        visualizer=visualizer,
        config=config
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    try:
        for epoch in range(start_epoch, config['training']['epochs']):
            logging.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
            
            # Training
            train_loss = trainer.train_epoch(train_loader, epoch)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = trainer.validate(val_loader)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            if lr_scheduler:
                lr_scheduler.step()
            
            # Logging
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")
            
            if args.wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': current_lr
                })
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            if (epoch + 1) % config['training'].get('checkpoint_interval', 10) == 0 or is_best:
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': config
                }, checkpoint_path, is_best)
                
                # Generate samples for visualization
                if (epoch + 1) % config['training'].get('sample_interval', 20) == 0:
                    with torch.no_grad():
                        samples = ddpm.sample(
                            batch_size=16,
                            image_size=config['data']['image_size'],
                            channels=config['model']['in_channels']
                        )
                        visualizer.save_checkpoint_visualization(epoch, train_loss, samples)
            
            # Plot training curves
            if (epoch + 1) % config['training'].get('plot_interval', 10) == 0:
                visualizer.plot_training_curves(
                    train_losses, val_losses,
                    save_path=output_dir / 'training_curves.png'
                )
    
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    
    finally:
        # Final checkpoint
        final_checkpoint_path = output_dir / 'final_checkpoint.pth'
        save_checkpoint({
            'epoch': len(train_losses) - 1,
            'model_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses[-1] if train_losses else 0,
            'val_loss': val_losses[-1] if val_losses else 0,
            'config': config
        }, final_checkpoint_path)
        
        # Final visualization
        visualizer.plot_training_curves(
            train_losses, val_losses,
            save_path=output_dir / 'final_training_curves.png'
        )
        
        logging.info("Training completed!")
        logging.info(f"Best validation loss: {best_val_loss:.6f}")
        
        if args.wandb:
            wandb.finish()


if __name__ == '__main__':
    main()
