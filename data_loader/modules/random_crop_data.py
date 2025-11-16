import random
import cv2
import numpy as np


class EastRandomCropData():
    def __init__(self, size=(640, 640), max_tries=50, min_crop_side_ratio=0.1, 
                 require_original_image=False, keep_ratio=True):
        """
        Random crop data augmentation for text detection (EAST-style)
        
        Args:
            size: Target output size as (width, height)
            max_tries: Maximum number of attempts to find valid crop
            min_crop_side_ratio: Minimum ratio of crop side to original image side
            require_original_image: If True, may return original image without cropping
            keep_ratio: If True, maintain aspect ratio and pad; if False, resize directly
        """
        self.size = size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.require_original_image = require_original_image
        self.keep_ratio = keep_ratio

    def __call__(self, data: dict) -> dict:
        """
        Apply random crop to image and text polygons
        
        Args:
            data: Dictionary containing 'img', 'text_polys', 'texts', 'ignore_tags'
        
        Returns:
            dict: Modified data dictionary with cropped image and adjusted polygons
        """
        im = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']
        texts = data['texts']
        
        # Get all polygons that should be preserved (not ignored)
        all_care_polys = [text_polys[i] for i, tag in enumerate(ignore_tags) if not tag]
        
        # Calculate crop region
        crop_x, crop_y, crop_w, crop_h = self.crop_area(im, all_care_polys)
        
        # Crop and resize image while maintaining aspect ratio
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        
        if self.keep_ratio:
            # Maintain aspect ratio with padding
            if len(im.shape) == 3:
                padimg = np.zeros((self.size[1], self.size[0], im.shape[2]), im.dtype)
            else:
                padimg = np.zeros((self.size[1], self.size[0]), im.dtype)
            padimg[:h, :w] = cv2.resize(im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
            img = padimg
        else:
            # Direct resize without maintaining aspect ratio
            img = cv2.resize(im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], tuple(self.size))
        
        # Crop and adjust text polygons
        text_polys_crop = []
        ignore_tags_crop = []
        texts_crop = []
        
        for poly, text, tag in zip(text_polys, texts, ignore_tags):
            # Adjust polygon coordinates based on crop and scale
            poly = ((poly - (crop_x, crop_y)) * scale).tolist()
            if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                text_polys_crop.append(poly)
                ignore_tags_crop.append(tag)
                texts_crop.append(text)
        
        data['img'] = img
        data['text_polys'] = np.float32(text_polys_crop)
        data['ignore_tags'] = ignore_tags_crop
        data['texts'] = texts_crop
        return data

    def is_poly_in_rect(self, poly, x, y, w, h):
        """
        Check if polygon is completely inside rectangle
        
        Args:
            poly: Polygon coordinates
            x, y: Top-left corner of rectangle
            w, h: Width and height of rectangle
        
        Returns:
            bool: True if polygon is completely inside rectangle
        """
        poly = np.array(poly)
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    def is_poly_outside_rect(self, poly, x, y, w, h):
        """
        Check if polygon is completely outside rectangle
        
        Args:
            poly: Polygon coordinates
            x, y: Top-left corner of rectangle
            w, h: Width and height of rectangle
        
        Returns:
            bool: True if polygon is completely outside rectangle
        """
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        """
        Split continuous regions in axis array
        
        Args:
            axis: Array of axis indices
        
        Returns:
            list: List of continuous regions
        """
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        """
        Randomly select two points from axis and return min/max
        
        Args:
            axis: Array of valid axis indices
            max_size: Maximum size to clip values
        
        Returns:
            tuple: (min_value, max_value)
        """
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions, max_size):
        """
        Randomly select points from different regions
        
        Args:
            regions: List of continuous regions
            max_size: Maximum size (not used but kept for consistency)
        
        Returns:
            tuple: (min_value, max_value)
        """
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax

    def crop_area(self, im, text_polys):
        """
        Find suitable crop area that doesn't split text instances
        
        Args:
            im: Input image
            text_polys: List of text polygon coordinates
        
        Returns:
            tuple: (crop_x, crop_y, crop_width, crop_height)
        """
        h, w = im.shape[:2]
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        
        # Mark regions occupied by text polygons
        for points in text_polys:
            points = np.round(points, decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        
        # Find regions without text (safe to crop)
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        # Split into continuous regions
        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        # Try to find valid crop region
        for i in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            # Check if crop area is large enough
            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                continue
            
            # Check if crop contains at least one polygon
            num_poly_in_rect = 0
            for poly in text_polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        # If no valid crop found, return full image
        return 0, 0, w, h


class PSERandomCrop():
    def __init__(self, size):
        """
        Random crop for PSE (Progressive Scale Expansion) text detection
        
        Args:
            size: Target crop size as (height, width)
        """
        self.size = size

    def __call__(self, data):
        """
        Apply random crop to multi-scale images (used in PSE)
        
        Args:
            data: Dictionary containing 'imgs' - list of images at different scales
                  imgs[0]: original image
                  imgs[1]: shrink label map
                  imgs[2]: threshold label map
        
        Returns:
            dict: Modified data dictionary with cropped images
        """
        imgs = data['imgs']

        h, w = imgs[0].shape[0:2]
        th, tw = self.size
        
        # If image size matches target, return as is
        if w == tw and h == th:
            return imgs

        # If text instances exist and random condition met, crop around text
        if np.max(imgs[2]) > 0 and random.random() > 3 / 8:
            # Find top-left corner of text instances
            tl = np.min(np.where(imgs[2] > 0), axis=1) - self.size
            tl[tl < 0] = 0
            
            # Find bottom-right corner of text instances
            br = np.max(np.where(imgs[2] > 0), axis=1) - self.size
            br[br < 0] = 0
            
            # Ensure enough space for crop when selecting bottom-right point
            br[0] = min(br[0], h - th)
            br[1] = min(br[1], w - tw)

            # Try to find valid crop containing text
            for _ in range(50000):
                i = random.randint(tl[0], br[0])
                j = random.randint(tl[1], br[1])
                # Ensure shrink_label_map contains text
                if imgs[1][i:i + th, j:j + tw].sum() <= 0:
                    continue
                else:
                    break
        else:
            # Random crop anywhere in image
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)

        # Crop all images in the list
        for idx in range(len(imgs)):
            if len(imgs[idx].shape) == 3:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
            else:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw]
        
        data['imgs'] = imgs
        return data