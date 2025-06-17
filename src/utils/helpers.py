import torch
import numpy as np
import os
import shutil
import json
import yaml
import random
from typing import Any, Dict, Union, Optional, Tuple
import logging
from PIL import Image
import torchvision.transforms as transforms


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False
):
    """
    Save model checkpoint.
    
    Args:
        state: Dictionary containing model state and metadata
        filepath: Path to save checkpoint
        is_best: Whether this is the best checkpoint so far
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save checkpoint
    torch.save(state, filepath)
    
    # Save best model copy
    if is_best:
        best_filepath = os.path.join(os.path.dirname(filepath), 'best_model.pth')
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(
    filepath: str,
    device: Union[str, torch.device] = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        device: Device to load checkpoint on
        
    Returns:
        Dictionary containing checkpoint data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.endswith('.json'):
            json.dump(config, f, indent=2)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy array.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Numpy array
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


def numpy_to_tensor(
    array: np.ndarray,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Convert numpy array to tensor.
    
    Args:
        array: Input numpy array
        device: Target device
        
    Returns:
        PyTorch tensor
    """
    tensor = torch.from_numpy(array)
    return tensor.to(device)


def normalize_tensor(
    tensor: torch.Tensor,
    mean: Optional[Union[float, list]] = None,
    std: Optional[Union[float, list]] = None
) -> torch.Tensor:
    """
    Normalize tensor with given mean and std.
    
    Args:
        tensor: Input tensor
        mean: Mean for normalization
        std: Standard deviation for normalization
        
    Returns:
        Normalized tensor
    """
    if mean is None:
        mean = tensor.mean()
    if std is None:
        std = tensor.std()
    
    if isinstance(mean, (list, tuple)):
        mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    if isinstance(std, (list, tuple)):
        std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    
    return (tensor - mean) / std


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Union[float, list],
    std: Union[float, list]
) -> torch.Tensor:
    """
    Denormalize tensor with given mean and std.
    
    Args:
        tensor: Normalized tensor
        mean: Mean used for normalization
        std: Standard deviation used for normalization
        
    Returns:
        Denormalized tensor
    """
    if isinstance(mean, (list, tuple)):
        mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    if isinstance(std, (list, tuple)):
        std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    
    return tensor * std + mean


def clamp_tensor(
    tensor: torch.Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0
) -> torch.Tensor:
    """
    Clamp tensor values to specified range.
    
    Args:
        tensor: Input tensor
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped tensor
    """
    return torch.clamp(tensor, min_val, max_val)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert tensor to PIL Image.
    
    Args:
        tensor: Image tensor (C, H, W) or (B, C, H, W)
        
    Returns:
        PIL Image
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image in batch
    
    # Move to CPU and detach
    tensor = tensor.detach().cpu()
    
    # Normalize to [0, 1] if needed
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    # Convert to PIL
    transform = transforms.ToPILImage()
    return transform(tensor)


def pil_to_tensor(
    image: Image.Image,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Convert PIL Image to tensor.
    
    Args:
        image: PIL Image
        device: Target device
        
    Returns:
        Image tensor
    """
    transform = transforms.ToTensor()
    tensor = transform(image)
    return tensor.to(device)


def make_grid(
    tensors: torch.Tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[float, float]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0
) -> torch.Tensor:
    """
    Make a grid of images from a batch of tensors.
    
    Args:
        tensors: Batch of image tensors (B, C, H, W)
        nrow: Number of images in each row
        padding: Padding between images
        normalize: Whether to normalize images
        value_range: Range for normalization
        scale_each: Whether to scale each image individually
        pad_value: Value for padding
        
    Returns:
        Grid tensor
    """
    from torchvision.utils import make_grid as tv_make_grid
    
    return tv_make_grid(
        tensors,
        nrow=nrow,
        padding=padding,
        normalize=normalize,
        value_range=value_range,
        scale_each=scale_each,
        pad_value=pad_value
    )


def interpolate_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """
    Linear interpolation between two tensors.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        alpha: Interpolation factor (0 = tensor1, 1 = tensor2)
        
    Returns:
        Interpolated tensor
    """
    return (1 - alpha) * tensor1 + alpha * tensor2


def get_model_size(model: torch.nn.Module) -> float:
    """
    Get model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def setup_logging(
    log_file: Optional[str] = None,
    log_level: str = 'INFO'
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional log file path
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('DDPM')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """
    Create experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to created experiment directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    
    return exp_dir


def compute_fid_score(
    real_features: torch.Tensor,
    fake_features: torch.Tensor
) -> float:
    """
    Compute Fréchet Inception Distance (FID) score.
    
    Args:
        real_features: Features from real images
        fake_features: Features from generated images
        
    Returns:
        FID score
    """
    # Convert to numpy
    real_features = tensor_to_numpy(real_features)
    fake_features = tensor_to_numpy(fake_features)
    
    # Compute means and covariances
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Compute FID
    diff = mu_real - mu_fake
    covmean, _ = np.linalg.eigh(sigma_real.dot(sigma_fake))
    covmean = np.sqrt(np.maximum(covmean, 0))
    
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * np.diag(covmean))
    
    return float(fid)


def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]['lr']


def print_model_summary(model: torch.nn.Module, input_size: Tuple[int, ...]):
    """
    Print model summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (without batch dimension)
    """
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size(model)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    print(f"Input size: {input_size}")
    print("=" * 60)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        self.register()
    
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
