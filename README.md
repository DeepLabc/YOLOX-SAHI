# YOLOX-SAHI
this repo mainly uses YOLOX to perform batch inference/slice inference on high-resolution images with SAHI

[SAHI](https://github.com/obss/sahi) is useful for small target detection in high-resolution images, it can be simply understood as splitting the picture and then inferring. Note that the original SAHI is inferred one by one, so the inference stage is too slow, if your resources permit, you can use batch-inference to obtain all the detection results one time. The following is a comparison of the results.


|   Left   | Right |
|  ----  | ----  |
| image(1080x1920)  | YOLOX_inference |
| YOLOX_slice_inference  | YOLOX_batch_inference |

--------------------------------------------------

--------------------------------------------------
<img src="images/road.jpg" width="400" height="600"><img src="/output/t6.png" width="400" height="600"><img src="/output/t7.png" width="400" height="600"><img src="/output/t8.png" width="400" height="600"/>

As you can see, the effect of using SAHI is very effective.

## Inference time (RTX 2080Ti)
* use standard SAHI slicing prediction, I eliminated the original standard inference(see in sahi/predict.py) and only infere 6 images.
```
Performing prediction on 6 number of slices.
Slicing performed in 0.06964755058288574 seconds.
Prediction performed in 0.19621586799621582 seconds.
```

* use batch-inference,note that the time of the slice already includes cropping and resize operations,Prediction performed time(total time) maybe is an intuitive indicator.
```
Performing prediction on 6 number of slices.
Slicing performed in 0.08550357818603516 seconds.
Prediction performed in 0.09710860252380371 seconds.
```
Using batch inference is much faster!!!
* difference:
```
Image.crop() vs img_slice = img[slice_bbox[1]:slice_bbox[3],slice_bbox[0]:slice_bbox[2], :]
single_image_input vs batch_image_input
```

# Use
* For more information about the environment configuration, see [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) Environment Configuration in the Reference section(I was already using YOLOX)

* Direct installation using "pip install -U yolox" might be effective[use anaconda(optional)]

* clone this repo
```
git clone https://github.com/DeepLabc/YOLOX-SAHI.git
```

```
python yolox_sahi.py
```
You can uncomment the corresponding script to get the results of other ways of reasoning

# Reference
* https://github.com/obss/sahi

* https://github.com/Resham-Sundar/sahi-yolox

* https://github.com/Megvii-BaseDetection/YOLOX
