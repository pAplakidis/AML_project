import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils

from model import *
from utils import *

from comboloss import ComboLoss

class Trainer:
  def __init__(self, device, model, train_loader, val_loader, model_path, early_stop=False, writer_path=None):
    self.early_stop = early_stop
    self.model_path = model_path
    if not writer_path:
      writer_path = TB_PATH
    print("[+] Tensorboard output path:", writer_path)

    #self.writer = SummaryWriter(writer_path)
    self.writer = SummaryWriter('runs/')
    self.device = device
    print("[+] Device:", self.device)
    self.model = model.to(self.device)
    self.train_loader = train_loader
    self.val_loader = val_loader

  def save_checkpoint(state, path):
    torch.save(state, path)
    print("Checkpoint saved at", path)

  def train(self, epochs=100, lr=1e-1, path=None):
  #def train(self, epochs=1, lr=1e-1, path=None):
    self.model.train()
    # class_weights = torch.FloatTensor([0.3, 0.3, 0.2, 0.1, 0.1]).to(self.device)  # TODO: finetune class weights
    # loss_func = nn.CrossEntropyLoss(weight=class_weights)
    loss_func = ComboLoss()
    optim = torch.optim.Adam(self.model.parameters(), lr=lr)

    # tb_images = next(iter(self.train_loader))['image'].float().to(self.device)
    # tb_masks = next(iter(self.train_loader))['mask'].float().to(self.device)
    # self.writer.add_graph(self.model, tb_images)
    #self.writer.add_graph(self.model, tb_masks)

    def eval(val_losses, val_seg_losses, val_clf_losses, val_iou, train=False):
      print("[+] Evaluating ...")
      with torch.no_grad():
        try:
          self.model.eval()
          l_idx = 0
          for i_batch, sample_batched in enumerate((t:= tqdm(self.val_loader))):
            X = sample_batched['image'].float().to(self.device)
            Y = sample_batched['mask'].float().to(self.device)
            Y_idx = torch.argmax(Y, dim=1) # Y_idx is ground truth
            Y_class = sample_batched['label'].long().to(self.device)

            out = self.model(X)
            loss, seg_loss, class_loss = loss_func(out[0], out[1], Y_idx, Y_class)
            iou_acc = iou_coef(Y, out[0])

            if not train:
              self.writer.add_scalar('evaluation loss', loss.item(), l_idx)
              self.writer.add_scalar('evaluation seg loss', seg_loss.item(), l_idx)
              self.writer.add_scalar('evaluation clf loss', class_loss.item(), l_idx)
              self.writer.add_scalar('evaluation IOU', iou_acc, l_idx)

            val_losses.append(loss.item())
            val_seg_losses.append(seg_loss.item())
            val_clf_losses.append(class_loss.item())
            val_iou.append(iou_acc)

            t.set_description("%d/%d: Batch Loss: %.2f - SegLoss: %.2f - CLF Loss: %.2f - IOU: %.2f"%(i_batch+1, len(self.val_loader), loss.item(), seg_loss.item(), class_loss.item(), iou_acc))
            l_idx += 1

        except KeyboardInterrupt:
          print("[~] Evaluation stopped by user")
      print("[+] Evaluation Done")
      return val_losses, val_seg_losses, val_clf_losses, val_iou

    losses, seg_losses, clf_losses = [], [], []
    px_accuracies = []  # TODO: per class accuracies
    iou_accuracies = []
    vlosses, v_seg_losses, v_clf_losses = [], [], []

    try:
      print("[+] Training ...")
      l_idx = 0
      for epoch in range(epochs):
        self.model.train()
        print("[+] Epoch %d/%d"%(epoch+1,epochs))
        epoch_losses, epoch_seg_losses, epoch_clf_losses = [], [], []
        epoch_pxaccuracies = []
        epoch_iouacc = []
        epoch_vlosses, epoch_v_seg_losses, epoch_v_clf_losses = [], [], []
        epoch_viouacc = []

        for i_batch, sample_batched in enumerate((t := tqdm(self.train_loader))):
          #print(i_batch+1, "/", len(self.train_loader), sample_batched['image'].size(), sample_batched['mask'].size())

          X = sample_batched['image'].float().to(self.device)
          Y = sample_batched['mask'].float().to(self.device)
          Y_idx = torch.argmax(Y, dim=1)  # NOTE: probably correct since it extracts indices i.e. the class the specific pixel belongs to
          Y_class = sample_batched['label'].long().to(self.device)

          # forward to net
          optim.zero_grad()
          out = self.model(X)
          #print(out)
          #print(Y_idx)
          #print(out.shape)
          #print(Y_idx.shape)
          # BUG: adding pixel_acc increases loss for some reason
          loss, seg_loss, class_loss = loss_func(out[0], out[1], Y_idx, Y_class)
          #pixel_acc = (torch.argmax(out, dim=1) == Y_idx).float().mean()
          #pixel_acc = (out == Y).float().mean()
          iou_acc = iou_coef(Y, out[0])

          self.writer.add_scalar('training running loss', loss.item(), l_idx)
          self.writer.add_scalar('training running seg loss', seg_loss.item(), l_idx)
          self.writer.add_scalar('training running clf loss', class_loss.item(), l_idx)
          #self.writer.add_scalar('training running pixel accuracy', pixel_acc.item(), l_idx)
          self.writer.add_scalar('training running iou accuracy', iou_acc, l_idx)
          epoch_losses.append(loss.item())
          epoch_seg_losses.append(seg_loss.item())
          epoch_clf_losses.append(class_loss.item())
          #epoch_pxaccuracies.append(pixel_acc.item())
          epoch_iouacc.append(iou_acc)

          loss.backward()
          optim.step()

          t.set_description("%d/%d: Batch TLoss: %.2f - SegLoss: %.2f - CLF Loss: %.2f -  IOU: %.2f"
                            % (i_batch+1, len(self.train_loader), loss.item(), seg_loss.item(), class_loss.item(), iou_acc)) 
          l_idx += 1
          
        avg_epoch_loss = np.array(epoch_losses).mean()
        avg_epoch_seg_loss = np.array(epoch_seg_losses).mean()
        avg_epoch_clf_loss = np.array(epoch_clf_losses).mean()
        #avg_epoch_pxacc = np.array(epoch_pxaccuracies).mean()
        avg_epoch_iouacc = np.array(epoch_iouacc).mean()
        losses.append(avg_epoch_loss)
        seg_losses.append(avg_epoch_seg_loss)
        clf_losses.append(avg_epoch_clf_loss)
        #px_accuracies.append(avg_epoch_pxacc)
        iou_accuracies.append(avg_epoch_iouacc)
        print("[=>] Epoch average training loss: %.4f"%avg_epoch_loss)
        self.writer.add_scalar('training epoch avg loss', avg_epoch_loss, epoch)
        print("[=>] Epoch average training seg loss: %.4f"%avg_epoch_seg_loss)
        self.writer.add_scalar('training epoch avg seg loss', avg_epoch_seg_loss, epoch)
        print("[=>] Epoch average training clf loss: %.4f"%avg_epoch_clf_loss)
        self.writer.add_scalar('training epoch avg clf loss', avg_epoch_clf_loss, epoch)
        #print("[=>] Epoch average training pixel accuracy: %.4f"%avg_epoch_pxacc)
        #self.writer.add_scalar('training epoch avg loss', avg_epoch_pxacc, epoch)
        print("[=>] Epoch average training IOU accuracy: %.4f"%avg_epoch_iouacc)
        self.writer.add_scalar('training epoch avg IOU accuracy', avg_epoch_iouacc, epoch)

        # TODO: implement early stopping to avoid overfitting
        if self.early_stop:
          epoch_vlosses, epoch_v_seg_losses, epoch_v_clf_losses, epoch_viouacc = eval(epoch_vlosses, epoch_v_seg_losses, epoch_v_clf_losses, epoch_viouacc, train=True)

          avg_epoch_vloss = np.array(epoch_vlosses).mean()
          avg_epoch_v_seg_loss = np.array(epoch_v_seg_losses).mean()
          avg_epoch_v_clf_loss = np.array(epoch_v_clf_losses).mean()

          vlosses.append(avg_epoch_vloss)
          v_seg_losses.append(avg_epoch_v_seg_loss)
          v_clf_losses.append(avg_epoch_v_clf_loss)
        print()

    except KeyboardInterrupt:
      print("[~] Training stopped by user")
    print("[+] Training Done")
    save_model(self.model_path, self.model)

    # plot final training stats
    for idx, l in enumerate(losses):
      self.writer.add_scalar("final training loss", l, idx)
    #plt.plot(losses, label="train loss")

    if self.early_stop:
      #plt.plot(vlosses, label="val loss")
      pass

    for idx, acc in enumerate(iou_accuracies):
      self.writer.add_scalar("final training IOU accuracy", acc, idx)
    #plt.plot(iou_accuracies, label="train iou acc")
    #plt.show()

    val_losses, val_seg_losses, val_clf_losses = [], [], []
    val_iou = []
    val_losses, val_seg_losses, val_clf_losses, epoch_viouacc = eval(val_losses, val_seg_losses, val_clf_losses, val_iou)
    print("Average Evaluation Loss: %.4f"%(np.array(val_losses).mean()))
    print("Average Evaluation IOU: %.4f"%(np.array(val_iou).mean()))

    self.writer.close()
    return self.model
