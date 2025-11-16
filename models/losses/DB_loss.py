"""
DBNet Loss Function

Implements the loss function for Differentiable Binarization (DB) networks
for text detection, combining multiple loss components for training.
"""

from torch import nn

from models.losses.basic_loss import BalanceCrossEntropyLoss, MaskL1Loss, DiceLoss


class DBLoss(nn.Module):
    """
    Differentiable Binarization Loss
    
    Multi-component loss function for DBNet text detection:
    1. Shrink map loss (binary cross-entropy with OHEM)
    2. Threshold map loss (L1 loss)
    3. Binary map loss (Dice loss, optional)
    
    The loss balances between accurate text region detection (shrink map),
    adaptive thresholding (threshold map), and final binarization (binary map).
    
    Reference:
    "Real-time Scene Text Detection with Differentiable Binarization"
    https://arxiv.org/abs/1911.08947
    """
    
    def __init__(self, alpha=1.0, beta=10, ohem_ratio=3, reduction='mean', eps=1e-6):
        """
        Initialize DBNet loss function
        
        Args:
            alpha: Weight coefficient for binary_map loss (default: 1.0)
            beta: Weight coefficient for threshold_map loss (default: 10)
                  Higher beta emphasizes learning adaptive thresholds
            ohem_ratio: Ratio for Online Hard Example Mining (OHEM)
                       Balances positive/negative samples (default: 3)
            reduction: Method to aggregate batch losses
                      'mean': average loss across batch
                      'sum': sum loss across batch
            eps: Small constant for numerical stability (default: 1e-6)
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], "reduction must be in ['mean', 'sum']"
        
        self.alpha = alpha
        self.beta = beta
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction
        
        # Loss components
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        self.dice_loss = DiceLoss(eps=float(eps))
        self.l1_loss = MaskL1Loss(eps=float(eps))

    def forward(self, pred, batch):
        """
        Calculate total loss for DBNet predictions
        
        Args:
            pred: Model predictions tensor of shape (B, C, H, W) where:
                  - pred[:, 0, :, :]: Shrink map (probability map of text regions)
                  - pred[:, 1, :, :]: Threshold map (adaptive threshold values)
                  - pred[:, 2, :, :]: Binary map (final binarized output, optional)
            batch: Dictionary containing ground truth data:
                  - 'shrink_map': Ground truth shrink map
                  - 'shrink_mask': Valid region mask for shrink map
                  - 'threshold_map': Ground truth threshold map
                  - 'threshold_mask': Valid region mask for threshold map
        
        Returns:
            dict: Dictionary containing:
                  - 'loss_shrink_maps': Shrink map loss value
                  - 'loss_threshold_maps': Threshold map loss value
                  - 'loss_binary_maps': Binary map loss value (if applicable)
                  - 'loss': Total weighted loss
        """
        # Extract prediction maps from model output
        shrink_maps = pred[:, 0, :, :]        # Probability map of text regions
        threshold_maps = pred[:, 1, :, :]     # Adaptive threshold values
        binary_maps = pred[:, 2, :, :]        # Final binary output (if exists)

        # Loss 1: Shrink map loss (BCE with OHEM for handling class imbalance)
        loss_shrink_maps = self.bce_loss(
            shrink_maps, 
            batch['shrink_map'], 
            batch['shrink_mask']
        )
        
        # Loss 2: Threshold map loss (L1 for smooth threshold learning)
        loss_threshold_maps = self.l1_loss(
            threshold_maps, 
            batch['threshold_map'], 
            batch['threshold_mask']
        )
        
        # Store individual loss components
        metrics = dict(
            loss_shrink_maps=loss_shrink_maps,
            loss_threshold_maps=loss_threshold_maps
        )
        
        # Loss 3: Binary map loss (optional, Dice loss for segmentation quality)
        if pred.size()[1] > 2:
            loss_binary_maps = self.dice_loss(
                binary_maps, 
                batch['shrink_map'], 
                batch['shrink_mask']
            )
            metrics['loss_binary_maps'] = loss_binary_maps
            
            # Total weighted loss
            loss_all = (
                self.alpha * loss_shrink_maps + 
                self.beta * loss_threshold_maps + 
                loss_binary_maps
            )
            metrics['loss'] = loss_all
        else:
            # If no binary map, only use shrink map loss
            metrics['loss'] = loss_shrink_maps
        
        return metrics