"""
DBNet Model Evaluation Script

Evaluates trained DBNet models on validation/test datasets and computes
text detection metrics (Precision, Recall, F-measure) along with FPS.
"""

import os
import sys
import pathlib

# Setup Python path for module imports
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import argparse
import time
import torch
from tqdm.auto import tqdm


class EVAL():
    """
    Evaluation class for DBNet text detection models
    
    Loads a trained model checkpoint and evaluates it on validation data,
    computing detection metrics and inference speed.
    """
    
    def __init__(self, model_path, gpu_id=0):
        """
        Initialize evaluator with model checkpoint
        
        Args:
            model_path: Path to model checkpoint file (.pth)
            gpu_id: GPU device ID to use (None or int)
                   If None or GPU unavailable, uses CPU
        """
        from models import build_model
        from data_loader import get_dataloader
        from post_processing import get_post_processing
        from utils import get_metric
        
        # Setup device
        self.gpu_id = gpu_id
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False  # No need for pretrained weights during eval

        # Setup validation data loader
        self.validate_loader = get_dataloader(
            config['dataset']['validate'], 
            config['distributed']
        )

        # Build model and load weights
        self.model = build_model(config['arch'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)

        # Setup post-processing and metrics
        self.post_process = get_post_processing(config['post_processing'])
        self.metric_cls = get_metric(config['metric'])

    def eval(self):
        """
        Run evaluation on validation dataset
        
        Computes:
        - Detection metrics (Recall, Precision, F-measure)
        - Inference speed (FPS - Frames Per Second)
        
        Returns:
            tuple: (recall, precision, f-measure) average values
        """
        self.model.eval()
        # Note: torch.cuda.empty_cache() can speed up evaluation after training
        
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0
        
        # Iterate through validation dataset
        for i, batch in tqdm(enumerate(self.validate_loader), 
                            total=len(self.validate_loader), 
                            desc='Evaluating model'):
            with torch.no_grad():
                # Move data to GPU/CPU
                for key, value in batch.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                
                # Inference and timing
                start = time.time()
                preds = self.model(batch['img'])
                boxes, scores = self.post_process(
                    batch, 
                    preds, 
                    is_output_polygon=self.metric_cls.is_output_polygon
                )
                # DEBUG: inspect first few batches
                if i < 5:
                    b = 0  # first image in batch
                    num_gt = len(batch['text_polys'][b])
                    num_pred = len(boxes[b])
                    print(f"\n[DEBUG] sample {i}, gt={num_gt}, pred={num_pred}")
                    if num_pred > 0:
                        print(" first pred box:", boxes[b][0])
                        print(" first pred score:", scores[b][0])
                
                total_frame += batch['img'].size()[0]
                total_time += time.time() - start
                
                # Calculate metrics for this batch
                raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores))
                raw_metrics.append(raw_metric)
                # DEBUG: print per-image metrics for first few batches
                if i < 5:
                    # batch size is 1, so raw_metric is a list of length 1
                    m = raw_metric[0]

                    # m comes from DetectionIoUEvaluator.evaluate_image(...)
                    # keys: 'precision','recall','hmean','gtCare','detCare','detMatched', etc.
                    print(
                        f"[MDEBUG] batch {i}: "
                        f"recall={m['recall']:.4f}, "
                        f"precision={m['precision']:.4f}, "
                        f"hmean={m['hmean']:.4f}, "
                        f"gt={m['gtCare']}, "
                        f"det={m['detCare']}, "
                        f"detMatched={m['detMatched']}"
                    )
        
        # Aggregate metrics across all batches
        metrics = self.metric_cls.gather_measure(raw_metrics)
        
        # Print inference speed
        print('FPS: {:.2f}'.format(total_frame / total_time))
        
        return (
            metrics['recall'].avg, 
            metrics['precision'].avg, 
            metrics['fmeasure'].avg
        )


def init_args():
    """
    Initialize command-line argument parser
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='DBNet.pytorch - Model Evaluation Script'
    )
    parser.add_argument(
        '--model_path', 
        required=False,
        default='output/DBNet_resnet18_FPN_DBHead/checkpoint/1.pth',
        type=str,
        help='Path to trained model checkpoint file'
    )
    parser.add_argument(
        '--gpu_id',
        required=False,
        default=0,
        type=int,
        help='GPU device ID (use -1 for CPU)'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    Main evaluation entry point
    
    Usage:
        python eval.py --model_path path/to/checkpoint.pth --gpu_id 0
    """
    args = init_args()
    
    # Initialize evaluator
    evaluator = EVAL(args.model_path, gpu_id=args.gpu_id if args.gpu_id >= 0 else None)
    
    # Run evaluation
    recall, precision, fmeasure = evaluator.eval()
    
    # Print results
    print('\n' + '='*50)
    print('Evaluation Results:')
    print('='*50)
    print(f'Recall:    {recall:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'F-measure: {fmeasure:.4f}')
    print('='*50)