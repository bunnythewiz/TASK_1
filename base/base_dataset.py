import copy
import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from data_loader.modules import *

import warnings
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Remove PIL size limit
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')

# For OpenCV JPEG warnings
import cv2
cv2.setLogLevel(0)  # Suppress all OpenCV warnings


class BaseDataSet(Dataset):

    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None,
                 target_transform=None):
        assert img_mode in ['RGB', 'BRG', 'GRAY']
        self.ignore_tags = ignore_tags
        self.data_list = self.load_data(data_path)
        item_keys = ['img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags']
        for item in item_keys:
            assert item in self.data_list[0], 'data_list from load_data must contains {}'.format(item_keys)
        self.img_mode = img_mode
        self.filter_keys = filter_keys
        self.transform = transform
        self.target_transform = target_transform
        self._init_pre_processes(pre_processes)

    def _init_pre_processes(self, pre_processes):
        self.aug = []
        if pre_processes is not None:
            for aug in pre_processes:
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']
                if isinstance(args, dict):
                    cls = eval(aug['type'])(**args)
                else:
                    cls = eval(aug['type'])(args)
                self.aug.append(cls)

    def load_data(self, data_path: str) -> list:
        """
        Load custom dataset with structure:
        data_path/
        ├── train/
        │   ├── images/
        │   └── gt/
        └── test/
            ├── images/
            └── gt/
        
        Ground truth format: x1,y1,x2,y2,x3,y3,x4,y4,text
        
        :params data_path: Path to train or test folder (e.g., 'datasets/train')
        :return: List of dicts containing 'img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags'
        """
        data_list = []
        data_path = Path(data_path)
        
        # Determine paths based on directory structure
        # If data_path is 'datasets/train' or 'datasets/test'
        if (data_path / 'images').exists():
            img_folder = data_path / 'images'
            gt_folder = data_path / 'gt'
        # If data_path directly points to images folder
        elif data_path.name == 'images':
            img_folder = data_path
            gt_folder = data_path.parent / 'gt'
        else:
            raise ValueError(f"Invalid data_path structure: {data_path}")
        
        if not gt_folder.exists():
            raise ValueError(f"Ground truth folder not found: {gt_folder}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(img_folder.glob(f'*{ext}')))
        
        print(f"Loading {len(image_files)} images from {img_folder}")
        
        # Process each image
        for img_path in sorted(image_files):
            # Get corresponding ground truth file
            gt_file = gt_folder / (img_path.stem + '.txt')
            
            if not gt_file.exists():
                print(f"Warning: Ground truth not found for {img_path.name}, skipping...")
                continue
            
            # Parse ground truth file
            text_polys = []
            texts = []
            ignore_tags = []
            
            try:
                with open(gt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Parse line: x1,y1,x2,y2,x3,y3,x4,y4,text
                        parts = line.split(',')
                        
                        if len(parts) < 9:
                            continue
                        
                        # Extract 8 coordinates
                        try:
                            coords = [float(parts[i]) for i in range(8)]
                        except ValueError:
                            print(f"Warning: Invalid coordinates in {gt_file.name}, skipping line")
                            continue
                        
                        # Reshape to (4, 2) - 4 points with (x, y) coordinates
                        poly = np.array(coords).reshape(4, 2)
                        
                        # Extract text (everything after 8th comma, in case text contains commas)
                        text = ','.join(parts[8:]).strip()
                        
                        # Determine if should ignore this text region
                        ignore = False
                        if not text or text == '###':
                            ignore = True
                        
                        # Check against ignore_tags list
                        for ignore_tag in self.ignore_tags:
                            if ignore_tag in text:
                                ignore = True
                                break
                        
                        text_polys.append(poly)
                        texts.append(text)
                        ignore_tags.append(ignore)
                
                # Only add if there are valid text regions
                if len(text_polys) > 0:
                    data_list.append({
                        'img_path': str(img_path.absolute()),
                        'img_name': img_path.name,
                        'text_polys': np.array(text_polys, dtype=np.float32),
                        'texts': texts,
                        'ignore_tags': ignore_tags
                    })
            
            except Exception as e:
                print(f"Error processing {gt_file.name}: {str(e)}")
                continue
        
        print(f"Successfully loaded {len(data_list)} samples")
        return data_list

    def apply_pre_processes(self, data):
        for aug in self.aug:
            data = aug(data)
        return data

    def __getitem__(self, index):
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                data = copy.deepcopy(self.data_list[index])
                im = cv2.imread(data['img_path'], 1 if self.img_mode != 'GRAY' else 0)
                
                if im is None:
                    raise ValueError(f"Failed to load image: {data['img_path']}")
                
                if self.img_mode == 'RGB':
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                
                # Resize large images to max 1280
                h, w = im.shape[:2]
                max_dim = 1280
                
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    
                    if 'text_polys' in data and len(data['text_polys']) > 0:
                        data['text_polys'] = np.array(data['text_polys'], dtype=np.float32) * scale
                
                # PAD to square 1280x1280 for batching
                h, w = im.shape[:2]
                pad_h = max_dim - h
                pad_w = max_dim - w
                im = np.pad(im, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
                
                data['img'] = im
                data['shape'] = [im.shape[0], im.shape[1]]
                data = self.apply_pre_processes(data)

                if self.transform:
                    if isinstance(self.transform, list):
                        for t in self.transform:
                            data = t(data)
                    else:
                        data = self.transform(data)
                        
                data['text_polys'] = data['text_polys'].tolist()
                
                if len(self.filter_keys):
                    data_dict = {}
                    for k, v in data.items():
                        if k not in self.filter_keys:
                            data_dict[k] = v
                    if 'img' in data_dict:
                        img = data_dict['img'].transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)
                        img = img.astype(np.float32) / 255.0  # uint8 [0,255] -> float32 [0,1]
                        data_dict['img'] = img
                    return data_dict
                else:
                    if 'img' in data:
                        img = data['img'].transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)
                        img = img.astype(np.float32) / 255.0  # uint8 [0,255] -> float32 [0,1]
                        data['img'] = img
                    return data
                    
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError(f"Failed to load data after {max_retries} attempts. Last error: {str(e)[:200]}")
                index = np.random.randint(self.__len__())
        
        raise RuntimeError("Max retries exceeded")

    def __len__(self):
        return len(self.data_list)