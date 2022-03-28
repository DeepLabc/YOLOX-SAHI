import json
import os
from unicodedata import category
import cv2
import time
from PIL import Image
from sahi.slicing import slice_image,slice_coco

image_path = "./images/"

"""
compare Image.crop with array oporate
"""
t1 = time.time()
img1 = Image.open("./images/road.jpg").convert("RGB")
crop_image = img1.crop((500,500,1000,1000))
t2 = time.time()

img2 =  cv2.imread("./images/road.jpg")
cc = img2[500:1000,500:1000,:]

print((t2-t1)/(time.time()-t2))

"""
use sahi slice image
"""
# slice_image_result, num_total_invalid_segmentation = slice_image(
#     image=image_path,
#     output_file_name='',
#     output_dir='',
#     slice_height=256,
#     slice_width=256,
#     overlap_height_ratio=0.2,
#     overlap_width_ratio=0.2,
# )
"""
COCO slice
"""
# coco_dict, coco_path = slice_coco(
#     coco_annotation_file_path='./instances_train2017.json',
#     output_coco_annotation_file_name='sahi',
#     output_dir='./images/slice_img/',
#     image_dir = image_path,
#     slice_height=512,
#     slice_width=640,
#     overlap_height_ratio=0.1,
#     overlap_width_ratio=0.1,
# )

# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw

# with open("./images/sahi_coco.json",'r') as pf:
#     data = json.load(pf)

# f, axarr = plt.subplots(2,2,figsize=(13,13))
# img_id = 0
# for row in range(2):
#     for col in range(2):
#         img = Image.open(data["images"][img_id]["file_name"]).convert('RGBA')
#         for ann_ind in range(len(data["annotations"])):
#             # find annotations that belong the selected image
#             if data["annotations"][ann_ind]["image_id"] == data["images"][img_id]["id"]:
#                 # convert coco bbox to pil bbox
#                 xywh = data["annotations"][ann_ind]["bbox"]
#                 xyxy = [xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3]]
#                 # visualize bbox over image
#                 ImageDraw.Draw(img, 'RGBA').rectangle(xyxy, width=5)
#             axarr[row, col].imshow(img)
#         img_id += 1

# plt.show()