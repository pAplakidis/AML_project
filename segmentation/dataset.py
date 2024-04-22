#!/usr/bin/env python3
import os
import  matplotlib.pyplot as plt
import random
import numpy as np

from torch.utils.data import Dataset

from utils import *

class SegDataset(Dataset):
  def __init__(self, base_dir):
    self.base_dir = base_dir
    print("[+] Dataset base dir:", self.base_dir)

    images_path = os.path.join(self.base_dir, "images")
    self.images = sorted(os.listdir(images_path))
    for i in range(len(self.images)):
      self.images[i] = os.path.join(images_path, self.images[i])

    masks_path = os.path.join(self.base_dir, "masks")
    self.masks = sorted(os.listdir(masks_path))
    for i in range(len(self.masks)):
      self.masks[i] = os.path.join(masks_path, self.masks[i])

    assert len(self.images) == len(self.masks)

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    images = np.load(self.images[idx])
    masks = np.load(self.masks[idx])
    return {"images": images, "masks": masks}


if __name__ == "__main__":
  dataset = SegDataset(BASE_DIR_SEG)
  sample = dataset[0]

  images, masks = sample["images"], sample["masks"]
  print(images.shape, masks.shape)

  n_slice = random.randint(0, masks.shape[2]-1)
  masks_view = from_categorical(masks)

  plt.figure(figsize=(12, 8))

  plt.subplot(221)
  plt.imshow(images[:, :, n_slice, 0], cmap='gray')
  plt.title('Image')

  plt.subplot(222)
  plt.imshow(masks_view[:, :, n_slice])
  plt.title('Mask')
  plt.show()
