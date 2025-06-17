#!/usr/bin/env python3
"""
Sampling/generation script for DDPM model.
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Dict, Any

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model.unet import UNet
from src.model.diffusion import DDPM
from src.utils.scheduler import get_noise_scheduler
from src.utils.helpers import setup_logging, load_checkpoint
from src.utils.visualization import DDPMVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate samples from DDPM model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='samples',
                       help='Output directory for samples')
    parser.add_argument('--num-samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for sampling')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device
