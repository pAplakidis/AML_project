import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime

W = H = 128

BASE_DIR_TRAIN = "../data/Brain_Tumor_MRI_Dataset/Training"
BASE_DIR_TEST = "../data/Brain_Tumor_MRI_Dataset/Testing"

BASE_DIR_SEG = "../data/BraTS2020/train"
BASE_DIR_SEG_VAL = "../data/BraTS2020/eval"
BASE_DIR_SEG_TEST = "../data/BraTS2020/test"

TB_PATH = "runs/" + str(datetime.now())


# 1-hot encodes a tensor
def to_categorical(y, num_classes):
  return np.eye(num_classes, dtype='uint8')[y]

# TODO: test this (might be ignoring classes here... or maybe it's the model)
def from_categorical(encoded_array):
  return np.argmax(encoded_array, axis=-1)

def view_net_result(origin_img, pred_img, gt_img=None):
  mask = np.moveaxis(pred_img.cpu().numpy(), 0, -1)
  mask_view = from_categorical(mask)

  plt.figure(figsize=(12, 8))
  plt.subplots_adjust(hspace=0.5)

  plt.subplot(231)
  plt.imshow(origin_img[:, :, 0], cmap='gray')
  plt.title('Image flair')

  plt.subplot(232)
  plt.imshow(origin_img[:, :, 1], cmap='gray')
  plt.title('Image t1ce')

  plt.subplot(233)
  plt.imshow(origin_img[:, :, 2], cmap='gray')
  plt.title('Image t2')

  # plt.subplot(221)
  # plt.imshow(origin_img[:, :, 0], cmap='gray')
  # plt.title('Image')

  if gt_img is not None:
    gt = np.moveaxis(gt_img, 0, -1)
    gt = from_categorical(gt)

    plt.subplot(234)
    plt.imshow(gt)
    plt.title('GT Mask')

  plt.subplot(235)
  plt.imshow(mask_view)
  plt.title('Predicted Mask')

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

