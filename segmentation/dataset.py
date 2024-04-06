#!/usr/bin/env python3
import os
import cv2
import numpy as np
from medpy.io import load, save

import torch
from torch.utils.data import Dataset

from utils import *

class SegDataset(Dataset):
  def __init__(self, base_dir, classes):
    self.base_dir = base_dir
    self.classes = classes

    self.hgg_dir = base_dir + "HGG/"  # TODO: LGG data as well
    self.patients = [self.hgg_dir + path for path in os.listdir(self.hgg_dir)]

  def __len__(self):
    return len(self.patients)

  def __getitem__(self, idx):
    path = self.patients[idx]
    files = os.listdir(path)
    # TODO: flair is the image, seg is the mask
    original = load(os.path.join(path, files[2]))[0]
    seg = load(os.path.join(path, files[1]))[0]

    mn=np.mean(original)
    std=np.std(original)
    pos=round((min(np.where(seg==1)[-1])+max(np.where(seg==1)[-1]))/2)
    dev=np.std(list(range(original.shape[-1])))*0.08
    
    img=cv2.resize(original[:,:,pos+int(dev)], (224,224), interpolation = cv2.INTER_AREA )
    img=(img-mn)/std
    img=np.repeat(np.expand_dims(img,-1),3,-1)
    
    img=cv2.resize(original[:,:,pos-int(dev)], (224,224), interpolation = cv2.INTER_AREA )
    img=(img-mn)/std
    img=np.repeat(np.expand_dims(img,-1),3,-1)
    
    
    img=cv2.resize(original[:,:,pos], (224,224), interpolation = cv2.INTER_AREA )
    img=(img-mn)/std
    img=np.repeat(np.expand_dims(img,-1),3,-1)

    print(img.shape)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img


if __name__ == "__main__":
  dataset = SegDataset(BASE_DIR_SEG, [])
  data = dataset[0]
