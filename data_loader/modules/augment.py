# -*- coding: utf-8 -*-
# Data augmentation modules for text detection

import math
import numbers
import random

import cv2
import numpy as np
from skimage.util import random_noise


class RandomNoise:
    def __init__(self, random_rate):
        """
        Add random Gaussian noise to images
        
        Args:
            random_rate: Probability of applying noise (0.0 to 1.0)
        """
        self.random_rate = random_rate

    def __call__(self, data: dict):
        """
        Apply random noise to image
        
        Args:
            data: Dictionary containing 'img', 'text_polys', 'texts', 'ignore_tags'
        
        Returns:
            dict: Modified data dictionary with noisy image
        """
        if random.random() > self.random_rate:
            return data
        
        im = data['img']
        data['img'] = (random_noise(im, mode='gaussian', clip=True) * 255).astype(im.dtype)
        return data


class RandomScale:
    def __init__(self, scales, random_rate):
        """
        Randomly scale images and text polygons
        
        Args:
            scales: List or array of scale factors to choose from
            random_rate: Probability of applying scaling (0.0 to 1.0)
        """
        self.random_rate = random_rate
        self.scales = scales

    def __call__(self, data: dict) -> dict:
        """
        Randomly select a scale and apply to image and text boxes
        
        Args:
            data: Dictionary containing 'img', 'text_polys', 'texts', 'ignore_tags'
        
        Returns:
            dict: Modified data dictionary with scaled image and polygons
        """
        if random.random() > self.random_rate:
            return data
        
        im = data['img']
        text_polys = data['text_polys']

        tmp_text_polys = text_polys.copy()
        rd_scale = float(np.random.choice(self.scales))
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        tmp_text_polys *= rd_scale

        data['img'] = im
        data['text_polys'] = tmp_text_polys
        return data


class RandomRotateImgBox:
    def __init__(self, degrees, random_rate, same_size=False):
        """
        Randomly rotate images and text polygons
        
        Args:
            degrees: Rotation angle range (single number or tuple/list of [min, max])
            random_rate: Probability of applying rotation (0.0 to 1.0)
            same_size: If True, keep output same size as input; if False, expand to fit rotated image
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            degrees = (-degrees, degrees)
        elif isinstance(degrees, (list, tuple, np.ndarray)):
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            degrees = degrees
        else:
            raise Exception('degrees must be Number or list or tuple or np.ndarray')
        
        self.degrees = degrees
        self.same_size = same_size
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        Apply random rotation to image and text boxes
        
        Args:
            data: Dictionary containing 'img', 'text_polys', 'texts', 'ignore_tags'
        
        Returns:
            dict: Modified data dictionary with rotated image and polygons
        """
        if random.random() > self.random_rate:
            return data
        
        im = data['img']
        text_polys = data['text_polys']

        # Rotate image
        w = im.shape[1]
        h = im.shape[0]
        angle = np.random.uniform(self.degrees[0], self.degrees[1])

        if self.same_size:
            nw = w
            nh = h
        else:
            # Convert angle to radians
            rangle = np.deg2rad(angle)
            # Calculate new width and height after rotation
            nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w))
            nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w))
        
        # Create rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, 1)
        # Calculate offset from original center to new center
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # Update rotation matrix with offset
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # Apply affine transformation
        rot_img = cv2.warpAffine(im, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        # Transform bounding box coordinates
        # Apply rotation matrix to each corner point of the bounding boxes
        rot_text_polys = list()
        for bbox in text_polys:
            point1 = np.dot(rot_mat, np.array([bbox[0, 0], bbox[0, 1], 1]))
            point2 = np.dot(rot_mat, np.array([bbox[1, 0], bbox[1, 1], 1]))
            point3 = np.dot(rot_mat, np.array([bbox[2, 0], bbox[2, 1], 1]))
            point4 = np.dot(rot_mat, np.array([bbox[3, 0], bbox[3, 1], 1]))
            rot_text_polys.append([point1, point2, point3, point4])
        
        data['img'] = rot_img
        data['text_polys'] = np.array(rot_text_polys)
        return data


class RandomResize:
    def __init__(self, size, random_rate, keep_ratio=False):
        """
        Randomly resize images and text polygons
        
        Args:
            size: Target size (single number or [width, height])
            random_rate: Probability of applying resize (0.0 to 1.0)
            keep_ratio: If True, pad shorter side to maintain aspect ratio
        """
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError("If size is a single number, it must be positive.")
            size = (size, size)
        elif isinstance(size, (list, tuple, np.ndarray)):
            if len(size) != 2:
                raise ValueError("If size is a sequence, it must be of len 2.")
            size = (size[0], size[1])
        else:
            raise Exception('size must be Number or list or tuple or np.ndarray')
        
        self.size = size
        self.keep_ratio = keep_ratio
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        Resize image and text boxes to target size
        
        Args:
            data: Dictionary containing 'img', 'text_polys', 'texts', 'ignore_tags'
        
        Returns:
            dict: Modified data dictionary with resized image and polygons
        """
        if random.random() > self.random_rate:
            return data
        
        im = data['img']
        text_polys = data['text_polys']

        if self.keep_ratio:
            # Pad shorter side to match longer side
            h, w, c = im.shape
            max_h = max(h, self.size[0])
            max_w = max(w, self.size[1])
            im_padded = np.zeros((max_h, max_w, c), dtype=np.uint8)
            im_padded[:h, :w] = im.copy()
            im = im_padded
        
        text_polys = text_polys.astype(np.float32)
        h, w, _ = im.shape
        im = cv2.resize(im, self.size)
        w_scale = self.size[0] / float(w)
        h_scale = self.size[1] / float(h)
        text_polys[:, :, 0] *= w_scale
        text_polys[:, :, 1] *= h_scale

        data['img'] = im
        data['text_polys'] = text_polys
        return data


def resize_image(img, short_size):
    """
    Resize image so that shorter side matches short_size, maintaining aspect ratio
    Output dimensions are multiples of 32
    
    Args:
        img: Input image
        short_size: Target size for shorter side
    
    Returns:
        tuple: (resized_image, (width_scale, height_scale))
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
    return resized_img, (new_width / width, new_height / height)


class ResizeShortSize:
    def __init__(self, short_size, resize_text_polys=True):
        """
        Resize image to ensure short side is at least short_size
        
        Args:
            short_size: Minimum size for shorter side
            resize_text_polys: If True, also scale text polygons
        """
        self.short_size = short_size
        self.resize_text_polys = resize_text_polys

    def __call__(self, data: dict) -> dict:
        """
        Resize image and text boxes based on short side
        
        Args:
            data: Dictionary containing 'img', 'text_polys', 'texts', 'ignore_tags'
        
        Returns:
            dict: Modified data dictionary with resized image and polygons
        """
        im = data['img']
        text_polys = data['text_polys']

        h, w, _ = im.shape
        short_edge = min(h, w)
        if short_edge < self.short_size:
            # Ensure short side >= short_size
            scale = self.short_size / short_edge
            im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
            scale = (scale, scale)
            
            if self.resize_text_polys:
                text_polys[:, :, 0] *= scale[0]
                text_polys[:, :, 1] *= scale[1]

        data['img'] = im
        data['text_polys'] = text_polys
        return data


class HorizontalFlip:
    def __init__(self, random_rate):
        """
        Randomly flip images horizontally
        
        Args:
            random_rate: Probability of applying flip (0.0 to 1.0)
        """
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        Apply horizontal flip to image and text boxes
        
        Args:
            data: Dictionary containing 'img', 'text_polys', 'texts', 'ignore_tags'
        
        Returns:
            dict: Modified data dictionary with flipped image and polygons
        """
        if random.random() > self.random_rate:
            return data
        
        im = data['img']
        text_polys = data['text_polys']

        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 1)
        h, w, _ = flip_im.shape
        # Mirror x-coordinates
        flip_text_polys[:, :, 0] = w - flip_text_polys[:, :, 0]

        data['img'] = flip_im
        data['text_polys'] = flip_text_polys
        return data


class VerticalFlip:
    def __init__(self, random_rate):
        """
        Randomly flip images vertically
        
        Args:
            random_rate: Probability of applying flip (0.0 to 1.0)
        """
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        Apply vertical flip to image and text boxes
        
        Args:
            data: Dictionary containing 'img', 'text_polys', 'texts', 'ignore_tags'
        
        Returns:
            dict: Modified data dictionary with flipped image and polygons
        """
        if random.random() > self.random_rate:
            return data
        
        im = data['img']
        text_polys = data['text_polys']

        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 0)
        h, w, _ = flip_im.shape
        # Mirror y-coordinates
        flip_text_polys[:, :, 1] = h - flip_text_polys[:, :, 1]
        
        data['img'] = flip_im
        data['text_polys'] = flip_text_polys
        return data