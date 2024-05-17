#!/usr/bin/env python3
import os
import  matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils import *


class ClassificationDataset(Dataset):
  def __init__(self, base_dir):
    self.base_dir = base_dir

    self.classes= sorted(os.listdir(base_dir))
    print(self.classes)
    self.image_paths = []
    self.labels = []

    for idx, dirr in enumerate(sorted(os.listdir(base_dir))):
      for image_path in sorted(os.listdir(os.path.join(base_dir, dirr))):
        self.image_paths.append(os.path.join(base_dir, dirr, image_path))
        self.labels.append(idx)

    assert len(self.image_paths) == len(self.labels)

    self.transform = transforms.Compose([
      transforms.Resize((W, H)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0, 0, 0], std=[1/255, 1/255, 1/255])  # Normalize to [0, 1]
    ])
  
  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    img = self.transform(Image.open(self.image_paths[idx])).unsqueeze(0).float()
    return {"image": img, "label": self.labels[idx]}


if __name__ == "__main__":
  dataset = ClassificationDataset(BASE_DIR_TRAIN)
  print(len(dataset))
  data = dataset[0]
  image, label = data["image"], data["label"]
  print(label, "-", image)
  print(image.shape)
