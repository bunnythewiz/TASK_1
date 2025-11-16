"""
Utility Functions for DBNet

Collection of helper functions for file I/O, visualization, logging,
geometry operations, and data processing for text detection tasks.
"""

import json
import pathlib
import time
import os
import glob
from natsort import natsorted
import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_file_list(folder_path: str, p_postfix: list = None, sub_dir: bool = True) -> list:
    """
    Get list of files with specified extensions from a folder
    
    Uses os.walk and os.listdir which are currently faster than pathlib.
    
    Args:
        folder_path: Path to folder to search
        p_postfix: List of file extensions to include (e.g., ['.jpg', '.png'])
                  Use ['.*'] to return all files
        sub_dir: Whether to search subdirectories recursively
    
    Returns:
        list: Naturally sorted list of file paths matching criteria
    """
    assert os.path.exists(folder_path) and os.path.isdir(folder_path), \
        f"Folder does not exist: {folder_path}"
    
    if p_postfix is None:
        p_postfix = ['.jpg']
    if isinstance(p_postfix, str):
        p_postfix = [p_postfix]
    
    file_list = [
        x for x in glob.glob(folder_path + '/**/*.*', recursive=sub_dir) 
        if os.path.splitext(x)[-1] in p_postfix or '.*' in p_postfix
    ]
    return natsorted(file_list)


def setup_logger(log_file_path: str = None):
    """
    Setup logger for DBNet training/inference
    
    Args:
        log_file_path: Optional path to log file. If None, only console logging is enabled
    
    Returns:
        logging.Logger: Configured logger instance
    """
    import logging
    logging._warn_preinit_stderr = 0
    
    logger = logging.getLogger('DBNet.pytorch')
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler (optional)
    if log_file_path is not None:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    
    logger.setLevel(logging.DEBUG)
    return logger


def exe_time(func):
    """
    Decorator to measure function execution time
    
    Usage:
        @exe_time
        def my_function():
            ...
    
    Args:
        func: Function to measure
    
    Returns:
        Wrapped function that prints execution time
    """
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        print("{} cost {:.3f}s".format(func.__name__, time.time() - t0))
        return back
    return newFunc


def load(file_path: str):
    """
    Load data from file based on extension
    
    Supports: .txt, .json, .list files
    
    Args:
        file_path: Path to file to load
    
    Returns:
        Loaded data (list for .txt/.list, dict for .json)
    """
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': _load_txt, '.json': _load_json, '.list': _load_txt}
    assert file_path.suffix in func_dict, f"Unsupported file type: {file_path.suffix}"
    return func_dict[file_path.suffix](file_path)


def _load_txt(file_path: str):
    """
    Load text file and return list of lines
    
    Removes BOM characters and strips whitespace
    
    Args:
        file_path: Path to text file
    
    Returns:
        list: List of lines from file
    """
    with open(file_path, 'r', encoding='utf8') as f:
        content = [x.strip().strip('\ufeff').strip('\xef\xbb\xbf') for x in f.readlines()]
    return content


def _load_json(file_path: str):
    """
    Load JSON file
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        dict: Parsed JSON data
    """
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def save(data, file_path):
    """
    Save data to file based on extension
    
    Supports: .txt, .json files
    
    Args:
        data: Data to save (list for .txt, dict for .json)
        file_path: Path to save file
    """
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': _save_txt, '.json': _save_json}
    assert file_path.suffix in func_dict, f"Unsupported file type: {file_path.suffix}"
    return func_dict[file_path.suffix](data, file_path)


def _save_txt(data, file_path):
    """
    Save list to text file (one item per line)
    
    Args:
        data: List of strings or single string to save
        file_path: Path to save file
    """
    if not isinstance(data, list):
        data = [data]
    with open(file_path, mode='w', encoding='utf8') as f:
        f.write('\n'.join(data))


def _save_json(data, file_path):
    """
    Save data to JSON file with formatting
    
    Args:
        data: Dictionary or list to save as JSON
        file_path: Path to save file
    """
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def show_img(imgs: np.ndarray, title='img'):
    """
    Display images using matplotlib
    
    Args:
        imgs: Image array (H, W) or (H, W, C)
        title: Title prefix for plot window
    """
    color = (len(imgs.shape) == 3 and imgs.shape[-1] == 3)
    imgs = np.expand_dims(imgs, axis=0)
    for i, img in enumerate(imgs):
        plt.figure()
        plt.title('{}_{}'.format(title, i))
        plt.imshow(img, cmap=None if color else 'gray')
    plt.show()


def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    """
    Draw bounding boxes on image
    
    Args:
        img_path: Path to image file or image array
        result: List of polygon coordinates (each as Nx2 array)
        color: BGR color tuple for drawing (default: red)
        thickness: Line thickness (default: 2)
    
    Returns:
        np.ndarray: Image with drawn bounding boxes
    """
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
    img_path = img_path.copy()
    
    for point in result:
        point = point.astype(int)
        cv2.polylines(img_path, [point], True, color, thickness)
    
    return img_path


def cal_text_score(texts, gt_texts, training_masks, running_metric_text, thred=0.5):
    """
    Calculate text detection scores using running metrics
    
    Args:
        texts: Predicted text probability maps
        gt_texts: Ground truth text maps
        training_masks: Valid region masks
        running_metric_text: Metric accumulator object
        thred: Threshold for binarization (default: 0.5)
    
    Returns:
        Score dictionary from running metric
    """
    training_masks = training_masks.data.cpu().numpy()
    pred_text = texts.data.cpu().numpy() * training_masks
    pred_text[pred_text <= thred] = 0
    pred_text[pred_text > thred] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text


def order_points_clockwise(pts):
    """
    Order 4 points in clockwise order: top-left, top-right, bottom-right, bottom-left
    
    Args:
        pts: Array of 4 points with shape (4, 2)
    
    Returns:
        np.ndarray: Reordered points array
    """
    rect = np.zeros((4, 2), dtype="float32")
    # Top-left point has smallest sum (x+y)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point has largest sum
    rect[2] = pts[np.argmax(s)]
    # Top-right has smallest difference (y-x)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left has largest difference
    rect[3] = pts[np.argmax(diff)]
    return rect


def order_points_clockwise_list(pts):
    """
    Alternative method to order points clockwise
    
    Sorts by y-coordinate first, then by x-coordinate
    
    Args:
        pts: Array of 4 points
    
    Returns:
        np.ndarray: Reordered points array
    """
    pts = pts.tolist()
    pts.sort(key=lambda x: (x[1], x[0]))
    pts[:2] = sorted(pts[:2], key=lambda x: x[0])
    pts[2:] = sorted(pts[2:], key=lambda x: -x[0])
    pts = np.array(pts)
    return pts


def get_datalist(folder_path) -> list:
    """
    Read train.txt or test.txt and return list of [img_path, annotation_data]
    
    annotation_data can be:
    - A file path (old format): gt/X123.txt
    - JSON string (new format): [{"transcription": "text", "points": [[x,y]...}]
    """
    import os
    
    # Handle if folder_path is a list (from config)
    if isinstance(folder_path, list):
        folder_path = folder_path[0]
    
    # Convert to string if Path object
    folder_path = str(folder_path)
    
    # Look for train.txt or test.txt in the folder
    if os.path.isdir(folder_path):
        for filename in ['train.txt', 'test.txt']:
            filepath = os.path.join(folder_path, filename)
            if os.path.exists(filepath):
                break
        else:
            raise FileNotFoundError(f"No train.txt or test.txt found in {folder_path}")
    else:
        filepath = folder_path
    
    data_list = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')  # Split by TAB
                if len(parts) >= 2:
                    # Make image path absolute
                    img_path = os.path.join(os.path.dirname(filepath), parts[0])
                    
                    # Check if annotation is JSON (starts with '[') or file path
                    annotation_data = parts[1]
                    if not annotation_data.startswith('['):
                        # Old format: it's a file path, make it absolute
                        annotation_data = os.path.join(os.path.dirname(filepath), annotation_data)
                    # else: New format: it's JSON, keep as-is
                    
                    data_list.append([img_path, annotation_data])
    
    return data_list


def parse_config(config: dict) -> dict:
    """
    Parse configuration file with base template inheritance
    
    Allows configs to inherit from base configs using 'base' key
    
    Args:
        config: Configuration dictionary
    
    Returns:
        dict: Merged configuration with base templates applied
    """
    import anyconfig
    base_file_list = config.pop('base')
    base_config = {}
    for base_file in base_file_list:
        tmp_config = anyconfig.load(open(base_file, 'rb'))
        if 'base' in tmp_config:
            tmp_config = parse_config(tmp_config)
        anyconfig.merge(tmp_config, base_config)
        base_config = tmp_config
    anyconfig.merge(base_config, config)
    return base_config


def save_result(result_path, box_list, score_list, is_output_polygon):
    """
    Save detection results to text file
    
    Args:
        result_path: Path to save results
        box_list: List of detected boxes/polygons
        score_list: List of confidence scores
        is_output_polygon: If True, save polygon format; if False, save box format
    """
    with open(result_path, 'wt') as res:
        for i, box in enumerate(box_list):
            score = score_list[i]
            box = box.reshape(-1).tolist()
            result = ",".join([str(int(x)) for x in box])
            res.write(result + ',' + str(score) + "\n")


def expand_polygon(polygon):
    """
    Expand polygon for single-character text boxes
    
    Expands narrow boxes to make them more visible/detectable
    
    Args:
        polygon: Polygon coordinates as Nx2 array
    
    Returns:
        np.ndarray: Expanded polygon coordinates (4, 2)
    """
    (x, y), (w, h), angle = cv2.minAreaRect(np.float32(polygon))
    if angle < -45:
        w, h = h, w
        angle += 90
    # Expand width by adding height
    new_w = w + h
    box = ((x, y), (new_w, h), angle)
    points = cv2.boxPoints(box)
    return order_points_clockwise(points)


if __name__ == '__main__':
    """
    Test utility functions
    """
    img = np.zeros((1, 3, 640, 640))
    show_img(img[0][0])
    plt.show()