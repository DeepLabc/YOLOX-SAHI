from distutils.command.config import config
from tabnanny import verbose
import yolox
import time
import cv2
import os
from sahi.model import YoloXDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction, get_batch_prediction

from IPython.display import Image
from yolox.data.datasets import COCO_CLASSES

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

kpt_file = '(your path)/YOLOX/yolox_s.pth'  # YOLOX pretrained weight
config_path = 'yolox_config'


detection_model = YoloXDetectionModel(
    model_path=kpt_file,
    config_path = config_path,
    device="cuda:0",
    confidence_threshold=0.3,
    nms_threshold=0.5,
    image_size = (640,640),  
    classes=COCO_CLASSES
)
"""
standard YOLOX prediction
"""
# result = get_prediction("/home/lyg/workspace/YOLOX_Det/task1/road.jpg", detection_model)
# result.export_visuals(export_dir="./task1/")

"""
standard YOLOX_slice prediction
"""
# result = get_sliced_prediction(
#     "./images/road.jpg",
#     detection_model=detection_model,
#     slice_height = 800,
#     slice_width = 800,
#     overlap_height_ratio = 0,
#     overlap_width_ratio = 0,
#     verbose=2
# )
# result.export_visuals(export_dir="./output/")  # set save file path, default name you can modify in sahi/prediction.py
# # print(result.to_coco_annotations())

"""
standard YOLOX_batch prediction
"""
result = get_batch_prediction(
    "./images/road.jpg",
    detection_model=detection_model,
    slice_height = 800,
    slice_width = 800,
    overlap_height_ratio = 0,
    overlap_width_ratio = 0,
    verbose=2
)
result.export_visuals(export_dir="./output/")  # save
