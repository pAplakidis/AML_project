#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch

W = H = 224
BASE_DIR_SEG = "../data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/train"
BASE_DIR_SEG_TEST = "../data/BraTS2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/test"

TB_PATH = "runs/single_batch_overfit"


# 1-hot encodes a tensor
def to_categorical(y, num_classes):
  return np.eye(num_classes, dtype='uint8')[y]

# TODO: test this (might be ignoring classes)
def from_categorical(encoded_array):
  return np.argmax(encoded_array, axis=-1)

def view_net_result(origin_img, pred_img, classes, gt_img=None):
  mask = np.moveaxis(pred_img.cpu().numpy(), 0, -1)
  mask_view = from_categorical(mask)

  plt.figure(figsize=(12, 8))

  plt.subplot(221)
  plt.imshow(origin_img[:, :, 0], cmap='gray')
  plt.title('Image')

  plt.subplot(223)
  plt.imshow(mask_view)
  plt.title('Predicted Mask')

  # TODO: test this as well
  if gt_img:
    # gt = segnet_to_rgb(torch.tensor(gt_img), classes)
    gt = gt_img.cpu().numpy()
    gt = np.moveaxis(gt, 0, -1)
    gt = from_categorical(gt)

    plt.subplot(222)
    plt.imshow(gt)
    plt.title('GT Mask')

  plt.show()

def get_only_object(img, mask, back_img):
  fg = cv2.bitwise_or(img, img, mask=mask)        
  # invert mask
  mask_inv = cv2.bitwise_not(mask)    
  fg_back_inv = cv2.bitwise_or(back_img, back_img, mask=mask_inv)
  final = cv2.bitwise_or(fg, fg_back_inv)
  return final

# intersect
def iou_coef(y_true, y_pred, smooth=1):
  intersection = torch.sum(torch.abs(y_true * y_pred), axis=[1,2,3])
  union = torch.sum(y_true, [1,2,3]) + torch.sum(y_pred, [1,2,3]) - intersection
  iou = torch.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou.item()

