"""
Dataset loading and preprocessing for DDPM training
"""

import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional, List, Union
import glob


class CustomImageDataset(Dataset):
    """
    Custom dataset for loading images from a directory
    """
    
    def __init__(
        self, 
        root_dir: str, 
        transform: Optional[transforms.Compose] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = extensions
        
        # Find all image files
        self.image_paths = []
        for ext in extensions:
            pattern = os.path.join(root_dir, '**', f'*{ext}')
            self.image_paths.extend(glob.glob(pattern, recursive=True))
            pattern = os.path.join(root_dir, '**', f'*{ext.upper()}')
            self.image_paths.extend(glob.glob(pattern, recursive=True))
            
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir} with extensions {extensions}")
            
        print(f"Found {len(self.image_paths)} images in {root_dir}")
        
    def __len__(self) -> int:
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random image if loading fails
            image = Image.new('RGB', (256, 256), color='white')
            
        if self.transform:
            image = self.transform(image)
            
        return image


def get_transform(
    image_size: int = 32,
    normalize: bool = True,
    horizontal_flip: bool = True,
    random_crop: bool = False,
    mean: List[float] = [0.5, 0.5, 0.5],
    std: List[float] = [0.5, 0.5, 0.5]
) -> transforms.Compose:
    """
    Get data transformation pipeline
    
    Args:
        image_size: Target image size
        normalize: Whether to normalize to [-1, 1]
        horizontal_flip: Whether to apply random horizontal flips
        random_crop: Whether to apply random cropping
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Transform composition
    """
    transform_list = []
    
    # Resize
    if random_crop:
        transform_list.append(transforms.Resize(int(image_size * 1.25)))
        transform_list.append(transforms.RandomCrop(image_size))
    else:
        transform_list.append(transforms.Resize((image_size, image_size)))
    
    # Data augmentation
    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize
    if normalize:
        transform_list.append(transforms.Normalize(mean, std))
        
    return transforms.Compose(transform_list)


def get_dataset(
    dataset_name: str,
    data_path: str = "./data",
    image_size: int = 32,
    train: bool = True,
    download: bool = True,
    **transform_kwargs
) -> Dataset:
    """
    Get dataset by name
    
    Args:
        dataset_name: Name of dataset ('cifar10', 'celeba', 'custom')
        data_path: Path to data directory
        image_size: Target image size
        train: Whether to get training set
        download: Whether to download dataset if not found
        **transform_kwargs: Additional transform arguments
        
    Returns:
        Dataset object
    """
    transform = get_transform(image_size=image_size, **transform_kwargs)
    
    if dataset_name.lower() == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=train,
            download=download,
            transform=transform
        )
    elif dataset_name.lower() == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root=data_path,
            train=train,
            download=download,
            transform=transform
        )
    elif dataset_name.lower() == 'mnist':
        # Convert MNIST to 3-channel for consistency
        mnist_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to 3-channel
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        dataset = torchvision.datasets.MNIST(
            root=data_path,
            train=train,
            download=download,
            transform=mnist_transform
        )
    elif dataset_name.lower() == 'celeba':
        try:
            dataset = torchvision.datasets.CelebA(
                root=data_path,
                split='train' if train else 'test',
                download=download,
                transform=transform
            )
        except:
            raise ValueError("CelebA dataset not available. Please download manually or use custom dataset.")
    elif dataset_name.lower() == 'custom':
        if not os.path.exists(data_path):
            raise ValueError(f"Custom dataset path {data_path} does not exist")
        dataset = CustomImageDataset(data_path, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return dataset


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True
) -> DataLoader:
    """
    Get data loader for dataset
    
    Args:
        dataset: Dataset object
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        DataLoader object
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def create_dataloaders(
    dataset_name: str,
    data_path: str = "./data",
    image_size: int = 32,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders
    
    Args:
        dataset_name: Name of dataset
        data_path: Path to data directory
        image_size: Target image size
        batch_size: Batch size
        num_workers: Number of workers
        pin_memory: Whether to pin memory
        **kwargs: Additional arguments for dataset creation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get training dataset
    train_dataset = get_dataset(
        dataset_name=dataset_name,
        data_path=data_path,
        image_size=image_size,
        train=True,
        **kwargs
    )
    
    train_loader = get_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    # Get validation dataset (if available)
    val_loader = None
    if dataset_name.lower() in ['cifar10', 'cifar100', 'mnist', 'celeba']:
        try:
            val_dataset = get_dataset(
                dataset_name=dataset_name,
                data_path=data_path,
                image_size=image_size,
                train=False,
                **kwargs
            )
            
            val_loader = get_dataloader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False
            )
        except:
            print("Validation dataset not available")
    
    return train_loader, val_loader


def unnormalize(tensor: torch.Tensor, mean: List[float] = [0.5, 0.5, 0.5], std: List[float] = [0.5, 0.5, 0.5]) -> torch.Tensor:
    """
    Unnormalize tensor from [-1, 1] to [0, 1]
    
    Args:
        tensor: Input tensor
        mean: Normalization mean that was used
        std: Normalization std that was used
        
    Returns:
        Unnormalized tensor
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert tensor to PIL Image
    
    Args:
        tensor: Input tensor of shape (C, H, W)
        
    Returns:
        PIL Image
    """
    # Unnormalize if needed
    if tensor.min() < 0:
        tensor = unnormalize(tensor.clone())
    
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and PIL
    tensor = tensor.cpu().numpy()
    tensor = (tensor * 255).astype('uint8')
    
    if tensor.shape[0] == 1:  # Grayscale
        tensor = tensor.squeeze(0)
        return Image.fromarray(tensor, mode='L')
    else:  # RGB
        tensor = tensor.transpose(1, 2, 0)
        return Image.fromarray(tensor, mode='RGB')
