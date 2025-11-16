# Data loader utilities for text detection training

import copy

import PIL
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def get_dataset(data_path, module_name, transform, dataset_args):
    """
    Get training dataset
    
    Args:
        data_path: Path to dataset folder or file containing image paths
        module_name: Custom dataset class name (e.g., 'BaseDataSet')
        transform: Transforms to apply to dataset
        dataset_args: Additional arguments for dataset initialization
    
    Returns:
        Dataset object if data_path is valid, otherwise None
    """
    from . import dataset
    s_dataset = getattr(dataset, module_name)(
        transform=transform, 
        data_path=data_path,
        **dataset_args
    )
    return s_dataset


def get_transforms(transforms_config):
    from data_loader import modules  # Import custom transforms
    import torchvision.transforms as torch_transforms
    
    tr_list = []
    for item in transforms_config:
        args = item.get('args', {})
        transform_type = item['type']
        
        # Try custom modules first, then torchvision
        if hasattr(modules, transform_type):
            cls = getattr(modules, transform_type)(**args)
        elif hasattr(torch_transforms, transform_type):
            cls = getattr(torch_transforms, transform_type)(**args)
        else:
            raise AttributeError(f"Transform '{transform_type}' not found in modules or torchvision.transforms")
        
        tr_list.append(cls)
    return tr_list


class ICDARCollectFN:
    """
    Custom collate function for ICDAR-style datasets
    Handles batching of samples with variable-length data
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize collate function"""
        pass

    def __call__(self, batch):
        """
        Collate batch samples into a single dictionary
        
        Args:
            batch: List of sample dictionaries from dataset
        
        Returns:
            dict: Batched data with tensors stacked where applicable
        """
        data_dict = {}
        to_tensor_keys = []
        
        # Collect all data into lists
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                # Track keys that contain tensor-like data
                if isinstance(v, (np.ndarray, torch.Tensor, PIL.Image.Image)):
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                data_dict[k].append(v)
        
        # Stack tensor-like data into batches
        for k in to_tensor_keys:
            data_dict[k] = torch.stack(data_dict[k], 0)
        
        return data_dict


def get_dataloader(module_config, distributed=False):
    """
    Create DataLoader from configuration
    
    Args:
        module_config: Configuration dictionary containing:
            - dataset: Dataset configuration
                - type: Dataset class name
                - args: Dataset arguments including data_path, transforms, etc.
            - loader: DataLoader configuration
                - batch_size: Batch size
                - num_workers: Number of data loading workers
                - shuffle: Whether to shuffle data
                - collate_fn: Custom collate function name (optional)
        distributed: If True, use DistributedSampler for multi-GPU training
    
    Returns:
        DataLoader object if configuration is valid, otherwise None
    """
    if module_config is None:
        return None
    
    config = copy.deepcopy(module_config)
    dataset_args = config['dataset']['args']
    
    # Get transforms if specified
    if 'transforms' in dataset_args:
        img_transforms = get_transforms(dataset_args.pop('transforms'))
    else:
        img_transforms = None
    
    # Create dataset
    dataset_name = config['dataset']['type']
    data_path = dataset_args.pop('data_path')
    
    if data_path is None:
        return None

    # Filter out None values from data_path list
    data_path = [x for x in data_path if x is not None]
    if len(data_path) == 0:
        return None
    
    # Setup collate function
    if 'collate_fn' not in config['loader'] or config['loader']['collate_fn'] is None or len(config['loader']['collate_fn']) == 0:
        config['loader']['collate_fn'] = None
    else:
        config['loader']['collate_fn'] = eval(config['loader']['collate_fn'])()

    # Create dataset instance
    _dataset = get_dataset(
        data_path=data_path, 
        module_name=dataset_name, 
        transform=img_transforms, 
        dataset_args=dataset_args
    )
    
    # Setup sampler for distributed training
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        # Use DistributedSampler for multi-GPU training
        sampler = DistributedSampler(_dataset)
        config['loader']['shuffle'] = False  # Sampler handles shuffling
        config['loader']['pin_memory'] = True  # Faster GPU transfer
    
    # Create DataLoader
    loader = DataLoader(dataset=_dataset, sampler=sampler, **config['loader'])
    return loader