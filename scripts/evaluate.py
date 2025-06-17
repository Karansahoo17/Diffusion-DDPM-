#!/usr/bin/env python3
"""
Evaluation script for DDPM model.
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Dict, Any, Tuple
import json

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model.unet import UNet
from src.model.diffusion import DDPM
from src.data.dataset import get_dataset
from src.utils.scheduler import get_noise_scheduler
from src.utils.helpers import setup_logging, load_checkpoint
from src.utils.visualization import DDPMVisualizer, create_comparison_plot
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate DDPM model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples for evaluation')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device to use')
    parser.add_argument('--compute-fid', action='store_true',
                       help='Compute FID score')
    parser.add_argument('--compute-is', action='store_true',
                       help='Compute Inception Score')
    parser.add_argument('--compute-lpips', action='store_true',
                       help='Compute LPIPS diversity score')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(gpu: int) -> torch.device:
    """Setup device for evaluation."""
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f'cuda:{gpu}')
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def load_model(checkpoint_path: str, config: Dict[str, Any], device: torch.device) -> DDPM:
    """Load trained DDPM model from checkpoint."""
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
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device)
    unet.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    ddpm.eval()
    
    return ddpm


def compute_reconstruction_loss(ddpm: DDPM, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Compute reconstruction loss on test data."""
    total_loss = 0.0
    timestep_losses = {}
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing reconstruction loss"):
            images = batch[0].to(device)
            batch_size = images.shape[0]
            
            # Sample random timesteps
            timesteps = torch.randint(0, ddpm.timesteps, (batch_size,), device=device)
            
            # Compute loss
            loss_dict = ddpm.compute_loss(images, timesteps, return_dict=True)
            
            total_loss += loss_dict['loss'].item() * batch_size
            num_samples += batch_size
            
            # Track losses per timestep
            for t, loss in zip(timesteps.cpu().numpy(), loss_dict['losses'].cpu().numpy()):
                if t not in timestep_losses:
                    timestep_losses[t] = []
                timestep_losses[t].append(loss)
    
    # Average timestep losses
    for t in timestep_losses:
        timestep_losses[t] = np.mean(timestep_losses[t])
    
    return {
        'avg_loss': total_loss / num_samples,
        'timestep_losses': timestep_losses
    }


def compute_sample_quality_metrics(samples: torch.Tensor) -> Dict[str, float]:
    """Compute basic quality metrics for generated samples."""
    samples_np = samples.cpu().numpy()
    
    return {
        'mean_pixel_value': np.mean(samples_np),
        'std_pixel_value': np.std(samples_np),
        'min_pixel_value': np.min(samples_np),
        'max_pixel_value': np.max(samples_np),
        'pixel_value_range': np.max(samples_np) - np.min(samples_np)
    }


def compute_fid_score(real_images: torch.Tensor, fake_images: torch.Tensor, device: torch.device) -> float:
    """Compute Fréchet Inception Distance (FID) score."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        
        # Ensure images are in [0, 1] range and have 3 channels
        real_images = (real_images + 1) / 2  # Convert from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2
        
        if real_images.shape[1] == 1:  # Grayscale to RGB
            real_images = real_images.repeat(1, 3, 1, 1)
        if fake_images.shape[1] == 1:
            fake_images = fake_images.repeat(1, 3, 1, 1)
        
        # Resize to 299x299 for Inception network
        real_images = torch.nn.functional.interpolate(real_images, size=(299, 299), mode='bilinear')
        fake_images = torch.nn.functional.interpolate(fake_images, size=(299, 299), mode='bilinear')
        
        # Update FID with real and fake images
        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)
        
        return fid.compute().item()
    
    except ImportError:
        logging.warning("torchmetrics not available. FID score computation skipped.")
        return -1.0


def compute_inception_score(samples: torch.Tensor, device: torch.device) -> Tuple[float, float]:
    """Compute Inception Score (IS)."""
    try:
        from torchmetrics.image.inception import InceptionScore
        
        inception_score = InceptionScore(normalize=True).to(device)
        
        # Ensure images are in [0, 1] range and have 3 channels
        samples = (samples + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        if samples.shape[1] == 1:  # Grayscale to RGB
            samples = samples.repeat(1, 3, 1, 1)
        
        # Resize to 299x299 for Inception network
        samples = torch.nn.functional.interpolate(samples, size=(299, 299), mode='bilinear')
        
        inception_score.update(samples)
        is_mean, is_std = inception_score.compute()
        
        return is_mean.item(), is_std.item()
    
    except ImportError:
        logging.warning("torchmetrics not available. IS computation skipped.")
        return -1.0, -1.0


def compute_lpips_diversity(samples: torch.Tensor, device: torch.device, num_pairs: int = 1000) -> float:
    """Compute LPIPS diversity score."""
    try:
        import lpips
        
        # Initialize LPIPS
        lpips_fn = lpips.LPIPS(net='alex').to(device)
        
        # Ensure images are in [-1, 1] range and have 3 channels
        if samples.shape[1] == 1:  # Grayscale to RGB
            samples = samples.repeat(1, 3, 1, 1)
        
        # Randomly sample pairs
        n_samples = samples.shape[0]
        indices1 = torch.randint(0, n_samples, (num_pairs,))
        indices2 = torch.randint(0, n_samples, (num_pairs,))
        
        # Ensure different samples
        mask = indices1 != indices2
        indices1 = indices1[mask]
        indices2 = indices2[mask]
        
        distances = []
        batch_size = 50  # Process in batches to avoid memory issues
        
        for i in range(0, len(indices1), batch_size):
            batch_indices1 = indices1[i:i + batch_size]
            batch_indices2 = indices2[i:i + batch_size]
            
            batch_samples1 = samples[batch_indices1]
            batch_samples2 = samples[batch_indices2]
            
            with torch.no_grad():
                batch_distances = lpips_fn(batch_samples1, batch_samples2)
                distances.extend(batch_distances.cpu().numpy())
        
        return np.mean(distances)
    
    except ImportError:
        logging.warning("lpips not available. LPIPS diversity computation skipped.")
        return -1.0


def evaluate_sampling_consistency(ddpm: DDPM, 
                                config: Dict[str, Any], 
                                num_runs: int = 5) -> Dict[str, float]:
    """Evaluate sampling consistency across multiple runs."""
    all_samples = []
    
    with torch.no_grad():
        for run in range(num_runs):
            samples = ddpm.sample(
                batch_size=16,
                image_size=config['data']['image_size'],
                channels=config['model']['in_channels']
            )
            all_samples.append(samples)
    
    # Compute statistics across runs
    all_samples = torch.stack(all_samples)  # [num_runs, batch_size, C, H, W]
    
    # Compute mean and std across runs for each sample position
    mean_across_runs = all_samples.mean(dim=0)
    std_across_runs = all_samples.std(dim=0)
    
    return {
        'mean_consistency': mean_across_runs.mean().item(),
        'std_consistency': std_across_runs.mean().item(),
        'max_std': std_across_runs.max().item(),
        'min_std': std_across_runs.min().item()
    }


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = setup_device(args.gpu)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup logging
    setup_logging(output_dir / 'evaluate.log')
    logging.info(f"Starting evaluation of checkpoint: {args.checkpoint}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model
    logging.info("Loading model...")
    ddpm = load_model(args.checkpoint, config, device)
    logging.info("Model loaded successfully")
    
    # Create visualizer
    visualizer = DDPMVisualizer(save_dir=output_dir)
    
    # Load test dataset
    logging.info("Loading test dataset...")
    test_dataset = get_dataset(
        dataset_name=config['data']['dataset'],
        data_dir=config['data']['data_dir'],
        image_size=config['data']['image_size'],
        train=False,
        transform_config=config['data'].get('transforms', {})
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logging.info(f"Test dataset size: {len(test_dataset)}")
    
    # Initialize results dictionary
    results = {}
    
    # 1. Compute reconstruction loss
    logging.info("Computing reconstruction loss...")
    loss_results = compute_reconstruction_loss(ddpm, test_loader, device)
    results['reconstruction_loss'] = loss_results['avg_loss']
    results['timestep_losses'] = loss_results['timestep_losses']
    
    # Visualize timestep losses
    visualizer.plot_timestep_loss_distribution(
        loss_results['timestep_losses'],
        save_path=output_dir / 'timestep_losses.png'
    )
    
    # 2. Generate samples for quality evaluation
    logging.info(f"Generating {args.num_samples} samples...")
    all_samples = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating samples"):
            current_batch_size = min(args.batch_size, args.num_samples - batch_idx * args.batch_size)
            
            samples = ddpm.sample(
                batch_size=current_batch_size,
                image_size=config['data']['image_size'],
                channels=config['model']['in_channels']
            )
            all_samples.append(samples)
    
    generated_samples = torch.cat(all_samples, dim=0)
    logging.info(f"Generated {generated_samples.shape[0]} samples")
    
    # 3. Compute basic quality metrics
    logging.info("Computing sample quality metrics...")
    quality_metrics = compute_sample_quality_metrics(generated_samples)
    results['quality_metrics'] = quality_metrics
    
    # 4. Visualize generated samples
    visualizer.plot_generated_samples(
        generated_samples[:64],
        n_rows=8,
        save_path=output_dir / 'generated_samples.png'
    )
    
    # 5. Compare with real samples
    logging.info("Creating comparison with real samples...")
    real_samples = []
    for batch in test_loader:
        real_samples.append(batch[0])
        if len(real_samples) * args.batch_size >= 64:
            break
    
    real_samples = torch.cat(real_samples, dim=0)[:64]
    create_comparison_plot(
        real_samples, generated_samples[:64],
        save_path=output_dir / 'real_vs_generated.png'
    )
    
    # 6. Compute FID score
    if args.compute_fid:
        logging.info("Computing FID score...")
        # Use subset for FID computation to save memory
        fid_samples = min(1000, len(real_samples), generated_samples.shape[0])
        fid_score = compute_fid_score(
            real_samples[:fid_samples], 
            generated_samples[:fid_samples], 
            device
        )
        results['fid_score'] = fid_score
        logging.info(f"FID Score: {fid_score:.4f}")
    
    # 7. Compute Inception Score
    if args.compute_is:
        logging.info("Computing Inception Score...")
        is_mean, is_std = compute_inception_score(generated_samples, device)
        results['inception_score'] = {'mean': is_mean, 'std': is_std}
        logging.info(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    
    # 8. Compute LPIPS diversity
    if args.compute_lpips:
        logging.info("Computing LPIPS diversity...")
        lpips_diversity = compute_lpips_diversity(generated_samples, device)
        results['lpips_diversity'] = lpips_diversity
        logging.info(f"LPIPS Diversity: {lpips_diversity:.4f}")
    
    # 9. Evaluate sampling consistency
    logging.info("Evaluating sampling consistency...")
    consistency_results = evaluate_sampling_consistency(ddpm, config)
    results['sampling_consistency'] = consistency_results
    
    # 10. Model analysis
    logging.info("Analyzing model...")
    total_params = sum(p.numel() for p in ddpm.model.parameters())
    trainable_params = sum(p.numel() for p in ddpm.model.parameters() if p.requires_grad)
    
    results['model_info'] = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }
    
    # Save results
    logging.info("Saving evaluation results...")
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Reconstruction Loss: {results['reconstruction_loss']:.6f}")
    print(f"Sample Quality - Mean: {quality_metrics['mean_pixel_value']:.4f}")
    print(f"Sample Quality - Std:  {quality_metrics['std_pixel_value']:.4f}")
    print(f"Model Parameters: {total_params:,}")
    
    if args.compute_fid and results.get('fid_score', -1) >= 0:
        print(f"FID Score: {results['fid_score']:.4f}")
    
    if args.compute_is and results.get('inception_score', {}).get('mean', -1) >= 0:
        print(f"Inception Score: {results['inception_score']['mean']:.4f} ± {results['inception_score']['std']:.4f}")
    
    if args.compute_lpips and results.get('lpips_diversity', -1) >= 0:
        print(f"LPIPS Diversity: {results['lpips_diversity']:.4f}")
    
    print(f"Sampling Consistency: {consistency_results['std_consistency']:.6f}")
    print("="*50)
    
    logging.info("Evaluation completed successfully!")
    logging.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
