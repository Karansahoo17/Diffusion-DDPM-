# DDPM: Denoising Diffusion Probabilistic Models

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) based on the paper "Denoising Diffusion Probabilistic Models" by Ho et al.

## Overview

This repository contains a complete implementation of DDPM for image generation. The model learns to generate images by learning to reverse a gradual noising process.This project implements the DDPM algorithm as described in the paper "Denoising Diffusion Probabilistic Models" by Ho et al. The implementation features a modular design with separate components for the U-Net architecture, diffusion process, training utilities, and evaluation tools.
Features

Modular Architecture: Clean separation of concerns with dedicated modules for model, training, data handling, and utilities
Flexible U-Net: Configurable U-Net architecture with attention mechanisms
Multiple Noise Schedules: Support for linear and cosine noise scheduling
Comprehensive Testing: Full test suite covering model components and diffusion process
Visualization Tools: Built-in utilities for visualizing training progress and generated samples
Easy Configuration: YAML-based configuration system

## Project Structure

```
ddpm/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── unet.py          # U-Net architecture for denoising
│   │   └── diffusion.py     # DDPM core logic
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py       # Data loading and preprocessing
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py       # Training loop and utilities
│   │   └── losses.py        # Loss functions
│   └── utils/
│       ├── __init__.py
│       ├── scheduler.py     # Noise scheduling
│       ├── helpers.py       # Utility functions
│       └── visualization.py # Plotting and visualization
├── scripts/
│   ├── train.py            # Training script
│   ├── sample.py           # Sampling/generation script
│   └── evaluate.py         # Evaluation script
├── notebooks/
│   └── demo.ipynb          # Jupyter notebook demo
└── tests/
    ├── __init__.py
    ├── test_model.py
    └── test_diffusion.py
```

## File Descriptions

### Core Model Files

- **`src/model/unet.py`**: Implements the U-Net architecture used as the denoising network. Includes time embedding, residual blocks, attention mechanisms, and skip connections.

- **`src/model/diffusion.py`**: Contains the core DDPM logic including forward diffusion (adding noise) and reverse diffusion (denoising) processes.

### Data and Training

- **`src/data/dataset.py`**: Handles data loading, preprocessing, and augmentation for training datasets.

- **`src/training/trainer.py`**: Main training loop with checkpoint saving, logging, and validation.

- **`src/training/losses.py`**: Loss functions including the simplified DDPM loss and optional variants.

### Utilities

- **`src/utils/scheduler.py`**: Implements various noise scheduling strategies (linear, cosine, etc.).

- **`src/utils/helpers.py`**: Utility functions for tensor operations, image processing, and model utilities.

- **`src/utils/visualization.py`**: Functions for visualizing training progress, generated samples, and loss curves.

### Scripts

- **`scripts/train.py`**: Main training script that can be run from command line with various arguments.

- **`scripts/sample.py`**: Script for generating samples from trained models.

- **`scripts/evaluate.py`**: Evaluation script for computing metrics like FID, IS, etc.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ddpm-pytorch.git
cd ddpm-pytorch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Training

Train a DDPM model on CIFAR-10:
```bash
python scripts/train.py --dataset cifar10 --epochs 1000 --batch_size 128 --lr 2e-4
```

Train on custom dataset:
```bash
python scripts/train.py --dataset custom --data_path /path/to/images --epochs 1000
```

### Sampling

Generate samples from a trained model:
```bash
python scripts/sample.py --model_path checkpoints/model_epoch_1000.pth --num_samples 64
```

### Configuration

Modify `config/config.yaml` to adjust model hyperparameters, training settings, and data paths.

## Model Architecture

The implementation uses a U-Net architecture with:
- Time embedding for diffusion timesteps
- Residual blocks with group normalization
- Self-attention mechanisms
- Skip connections between encoder and decoder
- Sinusoidal position embeddings

## Key Features

- **Flexible Architecture**: Easily configurable U-Net with different sizes and attention layers
- **Multiple Schedulers**: Support for linear, cosine, and other noise scheduling strategies
- **Checkpointing**: Automatic saving and loading of model checkpoints
- **Visualization**: Built-in tools for monitoring training and visualizing results
- **Extensible**: Clean, modular code structure for easy experimentation

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- numpy
- matplotlib
- tqdm
- PyYAML
- Pillow

## Results

The model can generate high-quality images after training. Sample results and training curves will be saved in the `results/` directory.

## Citation

If you use this implementation, please cite the original DDPM paper:

```bibtex
@article{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2006.11239},
  year={2020}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
