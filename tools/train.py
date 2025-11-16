"""
DBNet Training Script

Main training script for DBNet text detection model.
Supports single-GPU and multi-GPU distributed training.
"""

from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import argparse
import anyconfig


def init_args():
    """
    Initialize command-line argument parser
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='DBNet.pytorch - Text Detection Training'
    )
    parser.add_argument(
        '--config_file', 
        default='config/open_dataset_resnet18_FPN_DBhead_polyLR.yaml', 
        type=str,
        help='Path to training configuration YAML file'
    )
    parser.add_argument(
        '--local_rank', 
        dest='local_rank', 
        default=0, 
        type=int, 
        help='Local rank for distributed training (set automatically by torch.distributed.launch)'
    )

    args = parser.parse_args()
    return args


def main(config):
    """
    Main training function
    
    Sets up data loaders, model, loss function, metrics, and trainer,
    then starts the training process.
    
    Args:
        config: Configuration dictionary loaded from YAML file
    """
    import torch
    from models import build_model, build_loss
    from data_loader import get_dataloader
    from trainer import Trainer
    from post_processing import get_post_processing
    from utils import get_metric
    
    # Setup distributed training if multiple GPUs available
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl",  # NVIDIA Collective Communications Library
            init_method="env://",
            world_size=torch.cuda.device_count(),
            rank=args.local_rank
        )
        config['distributed'] = True
        print(f'Distributed training on {torch.cuda.device_count()} GPUs')
    else:
        config['distributed'] = False
        print('Single GPU training')
    
    config['local_rank'] = args.local_rank

    # Setup data loaders
    train_loader = get_dataloader(config['dataset']['train'], config['distributed'])
    assert train_loader is not None, 'Training data loader cannot be None'
    
    # Setup validation loader if specified in config
    if 'validate' in config['dataset']:
        validate_loader = get_dataloader(config['dataset']['validate'], False)
    else:
        validate_loader = None
        print('Warning: No validation dataset specified')

    # Build loss function and move to GPU
    criterion = build_loss(config['loss']).cuda()

    # Set input channels based on image mode (RGB: 3 channels, GRAY: 1 channel)
    config['arch']['backbone']['in_channels'] = (
        3 if config['dataset']['train']['dataset']['args']['img_mode'] != 'GRAY' else 1
    )
    
    # Build model
    model = build_model(config['arch'])

    # Setup post-processing for converting network output to text boxes
    post_p = get_post_processing(config['post_processing'])
    
    # Setup evaluation metrics
    metric = get_metric(config['metric'])

    # Initialize trainer
    trainer = Trainer(
        config=config,
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        post_process=post_p,
        metric_cls=metric,
        validate_loader=validate_loader
    )
    
    # Start training
    print('\n' + '='*50)
    print('Starting training...')
    print('='*50)
    trainer.train()


if __name__ == '__main__':
    """
    Entry point for training script
    
    Usage:
        # Single GPU training
        python train.py --config_file config/your_config.yaml
        
        # Multi-GPU distributed training
        python -m torch.distributed.launch --nproc_per_node=4 train.py --config_file config/your_config.yaml
    """
    import sys
    import pathlib
    
    # Setup Python path for module imports
    __dir__ = pathlib.Path(os.path.abspath(__file__))
    sys.path.append(str(__dir__))
    sys.path.append(str(__dir__.parent.parent))

    from utils import parse_config

    # Parse arguments
    args = init_args()
    
    # Load configuration file
    assert os.path.exists(args.config_file), f'Config file not found: {args.config_file}'
    
    print(f'Loading config from: {args.config_file}')
    config = anyconfig.load(open(args.config_file, 'rb'))
    
    # Parse config if it has a base template
    if 'base' in config:
        config = parse_config(config)
    
    # Print configuration summary
    print('\nConfiguration Summary:')
    print('-' * 50)
    print(f"Model: {config['arch']['type']}")
    print(f"Backbone: {config['arch']['backbone']['type']}")
    print(f"Dataset: {config['dataset']['train']['dataset']['args']['data_path']}")
    print(f"Batch size: {config['dataset']['train']['loader']['batch_size']}")
    print(f"Epochs: {config['trainer']['epochs']}")
    print(f"Learning rate: {config['optimizer']['args']['lr']}")
    print('-' * 50 + '\n')
    
    # Start training
    main(config)