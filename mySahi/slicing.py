# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import concurrent.futures
import logging
import os
import time
import torch
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image
from shapely.errors import TopologicalError
from tqdm import tqdm
from yolox.data.data_augment import ValTransform
from sahi.utils.cv import read_image_as_pil

logger = logging.getLogger(__name__)


def get_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: int = 0.2,
    overlap_width_ratio: int = 0.2,
):
    """Slices `image_pil` in crops.
    Corner values of each slice will be generated using the `slice_height`,
    `slice_width`, `overlap_height_ratio` and `overlap_width_ratio` arguments.

    Args:
        image_height (int): Height of the original image.
        image_width (int): Width of the original image.
        slice_height (int): Height of each slice. Default 512.
        slice_width (int): Width of each slice. Default 512.
        overlap_height_ratio(float): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio(float): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.

    Returns:
        List[List[int]]: List of 4 corner coordinates for each N slices.
            [
                [slice_0_left, slice_0_top, slice_0_right, slice_0_bottom],
                ...
                [slice_N_left, slice_N_top, slice_N_right, slice_N_bottom]
            ]
    """
    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes

def slice_image_batch(
    image: Union[str, Image.Image],
    output_file_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    min_area_ratio: float = 0.1,
    out_ext: Optional[str] = None,
    verbose: bool = False,
    infere_size = (640, 640),
):

    """Slice a large image into smaller windows. If output_file_name is given export
    sliced images.

    Args:
        image (str or PIL.Image): File path of image or Pillow Image to be sliced.
        coco_annotation_list (CocoAnnotation): List of CocoAnnotation objects.
        output_file_name (str, optional): Root name of output files (coordinates will
            be appended to this)
        output_dir (str, optional): Output directory
        slice_height (int): Height of each slice. Default 512.
        slice_width (int): Width of each slice. Default 512.
        overlap_height_ratio (float): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio (float): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        min_area_ratio (float): If the cropped annotation area to original annotation
            ratio is smaller than this value, the annotation is filtered out. Default 0.1.
        out_ext (str, optional): Extension of saved images. Default is the
            original suffix.
        verbose (bool, optional): Switch to print relevant values to screen.
            Default 'False'.

    Returns:
        sliced_image_result: SliceImageResult:
                                sliced_image_list: list of SlicedImage
                                image_dir: str
                                    Directory of the sliced image exports.
                                original_image_size: list of int
                                    Size of the unsliced original image in [height, width]
        num_total_invalid_segmentation: int
            Number of invalid segmentation annotations.
    """
    preproc = ValTransform(legacy=False)

    # read image
    img = cv2.imread(image)
    # img = read_image_as_pil(image)
   
    image_height, image_width, c = img.shape  #differece with Image.open().size
    # image_width, image_height = img.size
    if not (image_width != 0 and image_height != 0):
        raise RuntimeError(f"invalid image size: {img.size} for 'slice_image'.")
    
    slice_bboxes = get_slice_bboxes(
        image_height=image_height,
        image_width=image_width,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )
    slice_image_size = (slice_height, slice_width)
    # init images and annotations lists
    sliced_image_result = torch.empty((0,3,infere_size[0],infere_size[1]))

    # iterate over slices
    for slice_bbox in slice_bboxes:
        # extract image
        img_slice = img[slice_bbox[1]:slice_bbox[3],slice_bbox[0]:slice_bbox[2], :]
        img_pro, _ = preproc(img_slice, None, infere_size)  # 3, infere_size
      
        img_pro = torch.from_numpy(img_pro).unsqueeze(0)
        sliced_image_result = torch.cat([sliced_image_result,img_pro],0)
    
    return img, sliced_image_result, slice_bboxes, slice_image_size