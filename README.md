# 2022 GeoAI Martian Challenge

## 1. Result(Top1)
[official result](https://codalab.lisn.upsaclay.fr/competitions/1934#results)

| mAP50 | mAP@[.5,.95] | Rank |
|-------|--------------|------|
| 0.860 | 0.484        | 1    |


## 2. Experiments
| Model & tricks                            | mAP@[.5,.95] |
|-------------------------------------------|--------------|
| FasterRCNN                                | 0.424        |
| VFNet                                     | 0.460        |
| yolov5s                                   | 0.446        |
| yolov5s + DGIOU                           | 0.454        |
 | yolov5s + FNObj                           | 0.458        |
 | yolov5s + DGIOU + FNObj                   | 0.472        |
 | Ensemble(yolov5s + DGIOU + FNObj + VFNet) | 0.484        | 

## 3. How to reproduce?
All models are trained from scratch adn do not require fine-tuning. <br>

step1: Convert GeoAI dataset to the formats of mmdet and YOLO. <br>

step2: When training FasterRCNN and VFNet models using mmdet, it is important to note to use multi-scale training, replace nms with softnms, and set iou_thresh=0.5. please reference: mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py, line 108. <br>

step3: When training YOLO, different experimental results can be produced by modifying the loss configuration, including: <br>
1. yolov5/utils/loss.py, line 143, DGIOU Loss
2. yolov5/utils/loss.py, line 172, FNObj Loss

step4: Model Ensemble.
1. Perform inference on YOLOv5 and VFNet models separately, and save the inference results(bbox).
2. Calculate the performance(mAP) of YOLOv5 and VFNet on small, medium, and large scales on the validation set.
3. Run postprocess/fusion/merge.py to merge inference results(bbox) on test val(final result).




