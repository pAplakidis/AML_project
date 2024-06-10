# comboloss.py
import torch
import torch.nn as nn

#TODO: Initialize seg loss and clf loss in constructor

class ComboLoss(nn.Module):
    def __init__(self):
        super(ComboLoss, self).__init__()
        self.seg_loss_fn = nn.CrossEntropyLoss()
        self.class_loss_fn = nn.CrossEntropyLoss()

    def forward(self, seg_output, class_output, seg_target, class_target):
        seg_loss = self.seg_loss_fn(seg_output, seg_target)
        class_loss = self.class_loss_fn(class_output, class_target)
        total_loss = seg_loss + class_loss
        return total_loss, seg_loss, class_loss
