# -*- coding: utf-8 -*-
# Dataset implementations for various text detection datasets

import pathlib
import json
import os
import cv2
import numpy as np
import scipy.io as sio
from tqdm.auto import tqdm

from base import BaseDataSet
from utils import order_points_clockwise, get_datalist, load, expand_polygon


class ICDAR2015Dataset(BaseDataSet):
    """
    Dataset loader for ICDAR 2015 text detection competition
    """
    
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        """
        Initialize ICDAR 2015 dataset
        
        Args:
            data_path: Path to dataset folder
            img_mode: Image mode ('RGB', 'BRG', or 'GRAY')
            pre_processes: List of preprocessing operations
            filter_keys: Keys to filter from data
            ignore_tags: Tags to mark as ignored during training
            transform: Additional transforms to apply
        """
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        """
        Load ICDAR 2015 dataset annotations
        
        Args:
            data_path: Path to dataset folder
        
        Returns:
            list: List of data dictionaries containing image paths and annotations
        """
        data_list = get_datalist(data_path)
        t_data_list = []
        
        for img_path, annotation_data in data_list:
            data = self._get_annotation(annotation_data)
            if len(data['text_polys']) > 0:
                item = {
                    'img_path': img_path, 
                    'img_name': pathlib.Path(img_path).stem
                }
                item.update(data)
                t_data_list.append(item)
            else:
                print('No valid bounding boxes in {}'.format(annotation_data))
        
        return t_data_list

    def _get_annotation(self, annotation_data: str) -> dict:
        """
        Parse ICDAR 2015 annotation from inline JSON format.
        Format: [{"transcription": "text", "points": [[x1,y1], [x2,y2], ...]}, ...]
        """
        boxes = []
        texts = []
        ignores = []
        
        try:
            annotations = json.loads(annotation_data)
            for ann in annotations:
                points = np.array(ann['points'], dtype=np.float32)
                box = order_points_clockwise(points)
                
                if cv2.contourArea(box) > 0:
                    boxes.append(box)
                    texts.append(ann.get('transcription', ''))
                    ignores.append(ann.get('transcription', '') in self.ignore_tags)
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f'Failed to parse JSON: {e}')
            print(f'Data: {annotation_data[:100]}...')
        
        return {
            'text_polys': np.array(boxes) if boxes else np.array([]),
            'texts': texts,
            'ignore_tags': ignores,
        }


class DetDataset(BaseDataSet):
    """
    General detection dataset loader supporting character-level annotations
    Loads data from JSON files with flexible annotation format
    """
    
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        """
        Initialize detection dataset
        
        Args:
            data_path: Path to dataset folder
            img_mode: Image mode ('RGB', 'BRG', or 'GRAY')
            pre_processes: List of preprocessing operations
            filter_keys: Keys to filter from data
            ignore_tags: Tags to mark as ignored during training
            transform: Additional transforms to apply
            **kwargs: Additional arguments
                - load_char_annotation: If True, load character-level annotations
                - expand_one_char: If True, expand polygons for single characters
        """
        self.load_char_annotation = kwargs['load_char_annotation']
        self.expand_one_char = kwargs['expand_one_char']
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        """
        Load dataset from JSON files containing text and character annotations
        
        Args:
            data_path: List of paths to JSON annotation files
        
        Returns:
            list: List of data dictionaries with annotations
        """
        data_list = []
        
        for path in data_path:
            content = load(path)
            for gt in tqdm(content['data_list'], desc='Reading file {}'.format(path)):
                img_path = os.path.join(content['data_root'], gt['img_name'])
                polygons = []
                texts = []
                illegibility_list = []
                language_list = []
                
                # Process text-level annotations
                for annotation in gt['annotations']:
                    if len(annotation['polygon']) == 0 or len(annotation['text']) == 0:
                        continue
                    
                    # Expand polygon if text is longer than 1 character
                    if len(annotation['text']) > 1 and self.expand_one_char:
                        annotation['polygon'] = expand_polygon(annotation['polygon'])
                    
                    polygons.append(annotation['polygon'])
                    texts.append(annotation['text'])
                    illegibility_list.append(annotation['illegibility'])
                    language_list.append(annotation['language'])
                    
                    # Process character-level annotations if enabled
                    if self.load_char_annotation:
                        for char_annotation in annotation['chars']:
                            if len(char_annotation['polygon']) == 0 or len(char_annotation['char']) == 0:
                                continue
                            polygons.append(char_annotation['polygon'])
                            texts.append(char_annotation['char'])
                            illegibility_list.append(char_annotation['illegibility'])
                            language_list.append(char_annotation['language'])
                
                data_list.append({
                    'img_path': img_path,
                    'img_name': gt['img_name'],
                    'text_polys': np.array(polygons),
                    'texts': texts,
                    'ignore_tags': illegibility_list
                })
        
        return data_list


class SynthTextDataset(BaseDataSet):
    """
    Dataset loader for SynthText synthetic text dataset
    Reads annotations from MATLAB .mat files
    """
    
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags=None, transform=None, **kwargs):
        """
        Initialize SynthText dataset
        
        Args:
            data_path: Path to SynthText dataset root folder
            img_mode: Image mode ('RGB', 'BRG', or 'GRAY')
            pre_processes: List of preprocessing operations
            filter_keys: Keys to filter from data
            ignore_tags: Tags to mark as ignored during training
            transform: Additional transforms to apply
        """
        self.transform = transform
        self.dataRoot = pathlib.Path(data_path)
        
        if not self.dataRoot.exists():
            raise FileNotFoundError('Dataset folder does not exist: {}'.format(data_path))

        self.targetFilePath = self.dataRoot / 'gt.mat'
        if not self.targetFilePath.exists():
            raise FileExistsError('Ground truth file does not exist: {}'.format(self.targetFilePath))
        
        # Load MATLAB annotation file
        targets = {}
        sio.loadmat(
            self.targetFilePath, 
            targets, 
            squeeze_me=True, 
            struct_as_record=False,
            variable_names=['imnames', 'wordBB', 'txt']
        )

        self.imageNames = targets['imnames']
        self.wordBBoxes = targets['wordBB']
        self.transcripts = targets['txt']
        
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        """
        Load SynthText dataset from MATLAB annotation file
        
        Args:
            data_path: Path to dataset root (not used, data loaded in __init__)
        
        Returns:
            list: List of data dictionaries with annotations
        """
        t_data_list = []
        
        for imageName, wordBBoxes, texts in zip(self.imageNames, self.wordBBoxes, self.transcripts):
            item = {}
            
            # Handle 2D bounding boxes by adding dimension
            wordBBoxes = np.expand_dims(wordBBoxes, axis=2) if (wordBBoxes.ndim == 2) else wordBBoxes
            _, _, numOfWords = wordBBoxes.shape
            
            # Reshape bounding boxes to (num_words, 4, 2) format
            text_polys = wordBBoxes.reshape([8, numOfWords], order='F').T  # num_words * 8
            text_polys = text_polys.reshape(numOfWords, 4, 2)  # num_words * 4 * 2
            
            # Extract transcripts (split multi-word lines)
            transcripts = [word for line in texts for word in line.split()]
            
            # Skip if number of boxes doesn't match number of transcripts
            if numOfWords != len(transcripts):
                continue
            
            item['img_path'] = str(self.dataRoot / imageName)
            item['img_name'] = (self.dataRoot / imageName).stem
            item['text_polys'] = text_polys
            item['texts'] = transcripts
            item['ignore_tags'] = [x in self.ignore_tags for x in transcripts]
            t_data_list.append(item)
        
        return t_data_list


if __name__ == '__main__':
    """
    Test dataset loading functionality
    """
    import torch
    import anyconfig
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from utils import parse_config, show_img, plt, draw_bbox

    # Load configuration
    config = anyconfig.load('config/icdar2015_resnet18_FPN_DBhead_polyLR.yaml')
    config = parse_config(config)
    dataset_args = config['dataset']['train']['dataset']['args']
    
    # Create dataset
    train_data = ICDAR2015Dataset(
        data_path=dataset_args.pop('data_path'),
        transform=transforms.ToTensor(),
        **dataset_args
    )
    
    # Create dataloader
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    
    # Iterate through data
    for i, data in enumerate(tqdm(train_loader)):
        # Uncomment to visualize:
        # img = data['img']
        # shrink_label = data['shrink_map']
        # threshold_label = data['threshold_map']
        # show_img(img[0].numpy().transpose(1, 2, 0), title='img')
        # show_img((shrink_label[0].to(torch.float)).numpy(), title='shrink_label')
        # show_img((threshold_label[0].to(torch.float)).numpy(), title='threshold_label')
        # img = draw_bbox(img[0].numpy().transpose(1, 2, 0), np.array(data['text_polys']))
        # show_img(img, title='draw_bbox')
        # plt.show()
        pass