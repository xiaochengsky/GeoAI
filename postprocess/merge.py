# -*- coding: utf-8 -*-
# @Time : 2022/12/10 下午4:59
# @Author : YANG.C
# @File : merge.py

import json
import os
import sys
from tqdm import tqdm
import copy
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch

iou_thres = 0.8
score_thres = 0.44


def bbox_iou(box1, box2, x1y1x2y2=True, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    b1_x1, b1_x2 = box1[0], box1[0] + box1[2]
    b1_y1, b1_y2 = box1[1], box1[1] + box1[3]
    b2_x1, b2_x2 = box2[0], box2[0] + box2[2]
    b2_y1, b2_y2 = box2[1], box2[1] + box2[3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    return iou


def evaluate(dt_path):
    # load detection results, can be either a json file or a Python dictionary
    if isinstance(dt_path, str):
        dt = json.load(open(dt_path))
    else:
        dt = dt_path
    assert type(dt) == dict, 'Detection result format is not correct.'

    ids = list(dt.keys())
    ids = [int(x) for x in ids]

    # convert result into COCO format
    dt_coco = []
    for idx, bboxes in dt.items():
        for box in bboxes:
            if not len(box) == 5:
                raise Exception("Prediction format should be [xmin, ymin, width, height, score].")
            dt_dic = {"image_id": int(idx), "category_id": 1, "bbox": box[:4], "score": box[-1]}
            dt_coco.append(dt_dic)

    # evaluation
    ann_type = "bbox"
    coco_gt = COCO('gt_eval.json')
    # coco_gt = COCO('/home/yc/PyCharmProjects/datasets/GeoAI/annotations/instances_val2017.json')
    coco_dt = coco_gt.loadRes(dt_coco)
    coco_eval = COCOeval(coco_gt, coco_dt, ann_type)
    coco_eval.params.imgIds = ids
    coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 10.5 ** 2], [10.5 ** 2, 50.5 ** 2], [50.5 ** 2, 1e5 ** 2]]
    # coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 10.5 ** 2], [10.5 ** 2, 25.5 ** 2],
    #                             [25.5 ** 2, 50.5 ** 2], [50.5 ** 2, 1e5 ** 2]]
    # coco_eval.params.areaRngLbl = ['all', 'small', 'sub_small', 'medium', 'large']
    print(coco_eval.params.areaRngLbl)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


def merge(vf, yolo):
    large = 50.5 ** 2
    all_large = 1675
    filter = 0
    cnts = 0
    del_cnts = 0
    iou_match_cnts = 0
    with open(vf, 'r') as f:
        vf = json.load(f)
    with open(yolo, 'r') as f:
        yolo = json.load(f)

    # for img_id, bboxes in yolo.items():
    #     delete = []
    #     for i in range(len(bboxes)):
    #         w, h, s = bboxes[i][2:]
    #         if w * h >= large:
    #             delete.append(i)
    #
    #     delete = delete[::-1]
    #     for d in delete:
    #         del yolo[img_id][d]
    #         del_cnts += 1
    #
    for img_id, bboxes in vf.items():
        if img_id not in yolo.keys():
            yolo[img_id] = bboxes
        else:
            for bbox in bboxes:
                w, h, s = bbox[2:]
                if w * h >= large:
                    if s <= score_thres:
                        filter += 1
                        continue
                    yolo_bboxes = yolo[img_id]
                    if len(yolo_bboxes) == 0:
                        yolo[img_id].append(bbox)
                        cnts += 1
                        continue
                    yolo_bboxes_tensor = torch.tensor(yolo_bboxes)[:, :4]
                    vf_bbox = torch.tensor(bbox)
                    iou = bbox_iou(vf_bbox, yolo_bboxes_tensor)
                    if torch.max(iou).item() > iou_thres:
                        index = torch.argmax(iou).item()
                        del yolo[img_id][index]
                        yolo[img_id].append(bbox)
                        cnts += 1
                        iou_match_cnts += 1
                    else:
                        yolo[img_id].append(bbox)
                        cnts += 1

    dst_path = 'sub_merge_yolov5m.json'
    with open(dst_path, 'w') as f:
        json.dump(yolo, f)
    print(f'cnts: {cnts}, filter: {filter}, delcnts: {del_cnts}, iou_match_cnts: {iou_match_cnts}')

    evaluate(dst_path)


if __name__ == '__main__':
    vf = sys.argv[1]
    yolo = sys.argv[2]
    merge(vf, yolo)
