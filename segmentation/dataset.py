#!/usr/bin/env python3
import os
import  matplotlib.pyplot as plt
import random
import numpy as np

from torch.utils.data import Dataset

from utils import *

class SegDataset(Dataset):
  def __init__(self, base_dir, test=False):
    self.base_dir = base_dir
    self.test = test
    print("[+] Dataset base dir:", self.base_dir)

    images_path = os.path.join(self.base_dir, "images")
    self.images = sorted(os.listdir(images_path))
    for i in range(len(self.images)):
      self.images[i] = os.path.join(images_path, self.images[i])

    if not self.test:
      masks_path = os.path.join(self.base_dir, "masks")
      self.masks = sorted(os.listdir(masks_path))
      for i in range(len(self.masks)):
        self.masks[i] = os.path.join(masks_path, self.masks[i])

      assert len(self.images) == len(self.masks)

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    image = np.load(self.images[idx])
    image = np.moveaxis(image, -1, 0)

    if not self.test:
      mask = np.load(self.masks[idx])
      mask = np.moveaxis(mask, -1, 0)

      return {"image": image, "mask": mask}
    else:
      return {"image": image}


if __name__ == "__main__":
  dataset = SegDataset(BASE_DIR_SEG)
  print(len(dataset))
  sample = dataset[0]

  image, mask = sample["image"], sample["mask"]
  print(image.shape, mask.shape)
  image = np.moveaxis(image, 0, -1)
  mask = np.moveaxis(mask, 0, -1)

  mask_view = from_categorical(mask)
  print(mask_view.shape)

  plt.figure(figsize=(12, 8))

  plt.subplot(221)
  plt.imshow(image[:, :, 0], cmap='gray')
  plt.title('Image')

  plt.subplot(222)
  plt.imshow(mask_view)
  plt.title('Mask')
  plt.show()
