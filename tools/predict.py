"""
DBNet PyTorch Inference Script

Performs text detection inference on images using trained DBNet models.
Supports batch processing, visualization, and result saving.
"""

import os
import sys
import pathlib

# Setup Python path for module imports
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import time
import cv2
import torch

from data_loader import get_transforms
from models import build_model
from post_processing import get_post_processing


def resize_image(img, short_size):
    """
    Resize image so that shorter side equals short_size, maintaining aspect ratio.
    Output dimensions are multiples of 32 for network compatibility.
    
    Args:
        img: Input image (numpy array)
        short_size: Target size for shorter side
    
    Returns:
        Resized image (numpy array)
    """
    height, width, _ = img.shape
    
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    
    # Round to nearest multiple of 32
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


class Pytorch_model:
    """
    PyTorch model wrapper for DBNet text detection inference
    
    Handles model loading, image preprocessing, inference, and post-processing
    """
    
    def __init__(self, model_path, post_p_thre=0.7, gpu_id=None):
        """
        Initialize PyTorch model for inference
        
        Args:
            model_path: Path to model checkpoint file (.pth)
            post_p_thre: Threshold for post-processing (default: 0.7)
                        Higher values give fewer but more confident detections
            gpu_id: GPU device ID to use (None for CPU, int for specific GPU)
        """
        self.gpu_id = gpu_id

        # Setup device
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('Device:', self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Build model
        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False  # No pretrained weights needed for inference
        self.model = build_model(config['arch'])
        
        # Setup post-processing
        self.post_process = get_post_processing(config['post_processing'])
        self.post_process.box_thresh = post_p_thre
        
        # Get image mode from config
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        
        # Load model weights
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Setup transforms (only ToTensor and Normalize for inference)
        self.transform = []
        for t in config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        self.transform = get_transforms(self.transform)

    def predict(self, img_path: str, is_output_polygon=False, short_size: int = 1024):
        """
        Perform text detection on input image
        
        Args:
            img_path: Path to input image file
            is_output_polygon: If True, output polygon coordinates; if False, output bounding boxes
            short_size: Target size for shorter side of image (default: 1024)
        
        Returns:
            tuple: (prediction_map, box_list, score_list, inference_time)
                - prediction_map: Probability map of text regions
                - box_list: List of detected text boxes/polygons
                - score_list: Confidence scores for each detection
                - inference_time: Time taken for inference (seconds)
        """
        assert os.path.exists(img_path), f'File does not exist: {img_path}'
        
        # Load image
        img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0)
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        img = resize_image(img, short_size)
        
        # Transform image: (H, W, C) -> (1, C, H, W)
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        batch = {'shape': [(h, w)]}
        
        # Inference
        with torch.no_grad():
            # Synchronize for accurate timing on GPU
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            
            start = time.time()
            preds = self.model(tensor)
            
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            
            # Post-processing to get boxes and scores
            box_list, score_list = self.post_process(
                batch, preds, is_output_polygon=is_output_polygon
            )
            box_list, score_list = box_list[0], score_list[0]
            
            # Filter out invalid boxes (all zeros)
            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [score_list[i] for i, v in enumerate(idx) if v]
                else:
                    idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
            
            inference_time = time.time() - start
        
        return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, inference_time


def save_deploy(model, input, save_path):
    """
    Save model for deployment using TorchScript
    
    Args:
        model: PyTorch model to save
        input: Example input tensor for tracing
        save_path: Path to save traced model
    """
    traced_script_model = torch.jit.trace(model, input)
    traced_script_model.save(save_path)


def init_args():
    """
    Initialize command-line argument parser
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    import argparse
    parser = argparse.ArgumentParser(description='DBNet.pytorch - Text Detection Inference')
    parser.add_argument(
        '--model_path', 
        default=r'model_best.pth', 
        type=str,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--input_folder', 
        default='./test/input', 
        type=str, 
        help='Folder containing input images'
    )
    parser.add_argument(
        '--output_folder', 
        default='./test/output', 
        type=str, 
        help='Folder to save output results'
    )
    parser.add_argument(
        '--thre', 
        default=0.3, 
        type=float, 
        help='Threshold for post-processing (0.0-1.0)'
    )
    parser.add_argument(
        '--polygon', 
        action='store_true', 
        help='Output polygon coordinates instead of bounding boxes'
    )
    parser.add_argument(
        '--show', 
        action='store_true', 
        help='Display detection results using matplotlib'
    )
    parser.add_argument(
        '--save_result', 
        action='store_true', 
        help='Save detection boxes and scores to text file'
    )
    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int,
        help='GPU device ID (-1 for CPU)'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    Main inference script
    
    Usage:
        python predict.py --model_path model_best.pth --input_folder ./images --output_folder ./results
    """
    import pathlib
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from utils.util import show_img, draw_bbox, save_result, get_file_list

    args = init_args()
    print('Arguments:')
    print(args)
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id if args.gpu_id >= 0 else '0')
    
    # Initialize model
    print('\nInitializing model...')
    model = Pytorch_model(
        args.model_path, 
        post_p_thre=args.thre, 
        gpu_id=args.gpu_id if args.gpu_id >= 0 else None
    )
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Process images
    img_folder = pathlib.Path(args.input_folder)
    img_list = get_file_list(args.input_folder, p_postfix=['.jpg', '.png', '.jpeg'])
    
    print(f'\nProcessing {len(img_list)} images...')
    
    for img_path in tqdm(img_list, desc='Detecting text'):
        # Run inference
        preds, boxes_list, score_list, inference_time = model.predict(
            img_path, 
            is_output_polygon=args.polygon
        )
        
        # Draw bounding boxes on image
        img = draw_bbox(cv2.imread(img_path)[:, :, ::-1], boxes_list)
        
        # Display results if requested
        if args.show:
            show_img(preds, title='Probability Map')
            show_img(img, title=f'{os.path.basename(img_path)} ({inference_time:.3f}s)')
            plt.show()
        
        # Save results
        img_path = pathlib.Path(img_path)
        output_path = os.path.join(args.output_folder, img_path.stem + '_result.jpg')
        pred_path = os.path.join(args.output_folder, img_path.stem + '_pred.jpg')
        
        # Save detection result image
        cv2.imwrite(output_path, img[:, :, ::-1])
        # Save probability map
        cv2.imwrite(pred_path, preds * 255)
        
        # Save text file with box coordinates and scores
        if args.save_result:
            save_result(
                output_path.replace('_result.jpg', '.txt'), 
                boxes_list, 
                score_list, 
                args.polygon
            )
    
    print(f'\nResults saved to: {args.output_folder}')