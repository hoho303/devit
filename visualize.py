from pycocotools.coco import COCO
import numpy as np
import skimage.io as io

import matplotlib.pyplot as plt
import json
from PIL import Image
import torch
import cv2
from torchvision.ops import box_area, box_iou
import torchvision.ops as ops

dataset_root = 'Dataset/coco/'

coco2014_annFile = 'annotations/instances_val2014.json'
coco2014_imgDir = 'val2014/'

coco2014 = COCO(dataset_root + coco2014_annFile)

def convert_format_file(pred_path, out_path):
    preds = json.load(open(pred_path, 'r'))

    convert_pred = {}


    img_id = preds[0]['image_id']
    convert_preds = [{
        'bbox': preds[0]['bbox'],
        'score': preds[0]['score'],
        'category_id': preds[0]['category_id']
    }]

    i = 1
    while i < len(preds):
        pred = preds[i]
        if pred['image_id'] != img_id:
            convert_pred[img_id] = convert_preds
            convert_preds = []
            img_id = pred['image_id']
        
        convert_preds.append({
            'bbox': pred['bbox'],
            'score': pred['score'],
            'category_id': pred['category_id']
        })

        i += 1

    json.dump(convert_pred, open(out_path, 'w'))

def xywh2xyxy(bbox):
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    return bbox

def xyxy2xywh(bbox):
    bbox[2] -= bbox[0]
    bbox[3] -= bbox[1]
    return bbox
# def convert_format(preds):


def visualize(pred_path):
    preds = json.load(open(pred_path, 'r'))

    for key in preds.keys():
        img_id = int(key)
        img_info = coco2014.loadImgs(img_id)[0]

        img_path = dataset_root + coco2014_imgDir + img_info['file_name']
        print(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(len(preds[key]))
        for pred in preds[key]:
            pred_bbox = np.array(pred['bbox'], dtype=np.int32)
            img = cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]), (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 0), 2)
    
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    # pred_path = 'DeVIT/eval/few-shot/shot-30/vitl/inference/coco_instances_results.json'
    # out_path = 'devit/eval/few-shot/shot-10/vitl/inference/filter.json'
    out_path = 'devit/eval/few-shot/shot-10/vitl/inference/convert_format.json'
    # # convert_format_file(pred_path, out_path)
    visualize(out_path)