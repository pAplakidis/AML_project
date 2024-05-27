#!/usr/bin/env python3
import os
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from model import *
from dataset import *

MODEL_PATH = "./best_model_ENv2_torch_128x128.pth"

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[1/255, 1/255, 1/255])  # Normalize to [0, 1]
])


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)
  
  cl_dataset = ClassificationDataset(BASE_DIR_TRAIN)

  model = CustomEfficientNetV2(num_classes=len(cl_dataset.classes)).to(device)
  model = load_model(MODEL_PATH, model)

  # label training data
  train_labels = {}
  for image_name in (t:= tqdm(sorted(os.listdir(os.path.join(BASE_DIR_SEG, "images"))))):
    image = np.load(os.path.join(BASE_DIR_SEG, "images", image_name))
    image = test_transforms(image).unsqueeze(0).float().to(device)
    out = model(image)
    pred = torch.argmax(out).item()
    train_labels[image_name] = int(pred)
    t.set_description(cl_dataset.classes[pred])

  print(len(train_labels))
  train_labels_path = os.path.join(BASE_DIR_SEG, 'clf_labels.json')
  with open(train_labels_path, 'w') as f:
    json.dump(train_labels, f)
  print("[+] Train labels saved at:", train_labels_path)

  # label eval data
  eval_labels = {}
  for image_name in (t:= tqdm(sorted(os.listdir(os.path.join(BASE_DIR_SEG_VAL, "images"))))):
    image = np.load(os.path.join(BASE_DIR_SEG_VAL, "images", image_name))
    image = test_transforms(image).unsqueeze(0).float().to(device)
    out = model(image)
    pred = torch.argmax(out).item()
    eval_labels[image_name] = int(pred)
    t.set_description(cl_dataset.classes[pred])

  print(len(eval_labels))
  eval_labels_path = os.path.join(BASE_DIR_SEG_VAL, 'clf_labels.json')
  with open(eval_labels_path, 'w') as f:
    json.dump(train_labels, f)
  print("[+] Eval labels saved at:", eval_labels_path)
