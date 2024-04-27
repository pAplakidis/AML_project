#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

W = H = 224
BASE_DIR_SEG = "../data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/train"

TB_PATH = "runs/single_batch_overfit"


# 1-hot encodes a tensor
def to_categorical(y, num_classes):
  return np.eye(num_classes, dtype='uint8')[y]

# TODO: test this (might be ignoring classes)
def from_categorical(encoded_array):
  return np.argmax(encoded_array, axis=-1)


"""
CLASSES:
"""

def segnet_to_rgb(img, classes):
  # NOTE: 5,360,480 means that each channel of the 5 is a probability the pixel belongs in a specific class
  indices = torch.argmax(img, dim=0)

  out_img = []
  for i in range(indices.shape[0]):
    tmp = []
    for j in range(indices.shape[1]):
      tmp.append(classes[indices[i][j]])
    out_img.append(tmp)

  out_img = np.array(out_img, dtype=np.uint8)
  #print(out_img)
  #print(out_img.shape)
  return out_img


def overlay_mask(img, mask):
  """
  def to_png(img, a):
    fin_img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    b, g, r, alpha = cv2.split(fin_img)
    alpha = a
    fin_img[:,:, 0] = img[:,:,0]
    fin_img[:,:, 1] = img[:,:,1]
    fin_img[:,:, 2] = img[:,:,2]
    fin_img[:,:, 3] = alpha[:,:]
    return fin_img
  """

  #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # TODO: use the grayscale of this image
  for alpha in np.arange(0, 1.1, 0.1)[::-1]:
    overlay = img.copy()
    out = img.copy()
    overlay = cv2.bitwise_or(img, mask)

    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out

def view_net_result(origin_img, pred_img, classes, gt_img=None):
  cv2.imshow('image', origin_img)
  gt = segnet_to_rgb(torch.tensor(gt_img), classes)
  cv2.imshow('ground truth', gt)
  segmented_img = segnet_to_rgb(pred_img, classes)
  cv2.imshow('prediction', segmented_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def get_only_object(img, mask, back_img):
  fg = cv2.bitwise_or(img, img, mask=mask)        
  # invert mask
  mask_inv = cv2.bitwise_not(mask)    
  fg_back_inv = cv2.bitwise_or(back_img, back_img, mask=mask_inv)
  final = cv2.bitwise_or(fg, fg_back_inv)
  return final

# TODO: view network's output overlapping original image
def view_overlap_net_result(origin_img, pred_img, gt_img=None):
  pass

# intersect
def iou_coef(y_true, y_pred, smooth=1):
  intersection = torch.sum(torch.abs(y_true * y_pred), axis=[1,2,3])
  union = torch.sum(y_true, [1,2,3]) + torch.sum(y_pred, [1,2,3]) - intersection
  iou = torch.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou.item()

def show_imgmask_batch(sample_batched):
  img_batch, mask_batch = sample_batched['image'], sample_batched['mask']
  batch_size = len(img_batch)
  img_size = img_batch.size(2)
  grid_border_size = 2

  grid = utils.make_grid(img_batch)
  #grid = utils.make_grid(mask_batch)
  plt.imshow(grid.numpy().transpose((1,2,0)))

