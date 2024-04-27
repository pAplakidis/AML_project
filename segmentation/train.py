#!/usr/bin/env python3
import os
import torch
from torch.utils.data import DataLoader, random_split

from model import *
from dataset import SegDataset
from trainer import Trainer
from utils import *

# EXAMPLE RUN: EPOCHS=100 LR=1e-3 WRITER_PATH="runs/overfit" MODEL_PATH="models/segnet.pt" ./train.py

BS = os.getenv("BS")
if BS == None:
  BS = 32  # NOTE: this is the max batch size my home-PC can handle, paper used 12
else:
  BS = int(BS)
print("[+] Using Batch Size:", BS)

EPOCHS = os.getenv("EPOCHS")
if EPOCHS != None:
  EPOCHS = int(EPOCHS)
else:
  EPOCHS = 100
print("[+] Max epochs:", EPOCHS)

LR = os.getenv("EPOCHS")
if LR != None:
  LR = float(LR)
else:
  LR = 1e-3
print("[+] Learning Rate:", LR)

N_WORKERS = 8

MODEL_PATH = os.getenv("MODEL_PATH")
if MODEL_PATH == None:
  MODEL_PATH = "models/segnet.pt"
print("[+] Model save path:", MODEL_PATH)

# TODO: check out how it's done in ADA
writer_path = os.getenv("WRITER_PATH")


if __name__ == '__main__':
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  # get data
  dataset = SegDataset(BASE_DIR_SEG)
  train_set, val_set = random_split(dataset, [int(len(dataset)*0.7)+1, int(len(dataset)*0.3)])
  train_loader = DataLoader(train_set, batch_size=BS, shuffle=True, num_workers=N_WORKERS, pin_memory=True)
  val_loader = DataLoader(val_set, batch_size=BS, shuffle=True, num_workers=N_WORKERS, pin_memory=True)

  # define model and train
  in_samp = dataset[0]['image']
  out_samp = dataset[0]['mask']
  print("Image shape:", in_samp.shape, " - Mask shape:", out_samp.shape)
  in_ch, out_ch = in_samp.shape[0], out_samp.shape[0]
  model = SegNet(in_ch, out_ch)

  if writer_path:
    trainer = Trainer(device, model, train_loader, val_loader, MODEL_PATH, writer_path=writer_path)
  else:
    trainer = Trainer(device, model, train_loader, val_loader, MODEL_PATH)
    model = trainer.train(epochs=EPOCHS, lr=LR)  # NOTE: lr=1e-3 seems to be optimal
  # else:
  #   model = load_model(MODEL_PATH, model)
  #   model.to(device)

  # # view some images to examine the model's progress
  # with torch.no_grad():
  #   model.eval()
  #   print("[*] Training images preview")
  #   for i in range(5):
  #     samp = train_set[random.randint(0, len(train_set))]
  #     img, mask = samp['image'], samp['mask']
  #     out_img = np.moveaxis(img, 0, -1)
  #     X = torch.tensor([img, img]).float().to(device)
  #     pred = model(X)
  #     view_net_result(out_img, pred[0], classes, gt_img=mask)

  #   print("[*] Evaluation images preview")
  #   for i in range(5):
  #     samp = val_set[random.randint(0, len(val_set))]
  #     img, mask = samp['image'], samp['mask']
  #     out_img = np.moveaxis(img, 0, -1)
  #     X = torch.tensor([img, img]).float().to(device)
  #     pred = model(X)
  #     view_net_result(out_img, pred[0], classes, gt_img=mask)
