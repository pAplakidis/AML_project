#!/usr/bin/env python3
import os
import json
import  matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset

from utils import *

class MultitaskDataset(Dataset):
  def __init__(self, base_dir, test=False):
    self.base_dir = base_dir
    self.test = test
    print("[+] Dataset base dir:", self.base_dir)

    BASE_DIR_CLF = "../data/Brain_Tumor_MRI_Dataset/Training" # TODO: cleanup
    self.classes_path = os.path.join(base_dir, "clf_labels.json")
    self.classes= sorted(os.listdir(BASE_DIR_CLF))
    with open(self.classes_path) as f:
      self.clf_labels = json.load(f)

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
    image_name = self.images[idx].split('/')[-1]
    print(image_name)
    image = np.load(self.images[idx])
    image = np.moveaxis(image, -1, 0)

    if not self.test:
      mask = np.load(self.masks[idx])
      mask = np.moveaxis(mask, -1, 0)
      label = self.clf_labels[image_name]

      return {"image": image, "mask": mask, "label": label}
    else:
      return {"image": image}


if __name__ == "__main__":
  dataset = MultitaskDataset(BASE_DIR_SEG)
  print(len(dataset))
  sample = dataset[0]

  image, mask, label = sample["image"], sample["mask"], sample["label"]
  print(image.shape, mask.shape, label)
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

  plt.subplot(223)
  plt.title(dataset.classes[label])

  plt.show()

