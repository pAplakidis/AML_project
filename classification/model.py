import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class CustomEfficientNetV2(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNetV2, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')  # You can choose other versions like 'efficientnet-b1', 'efficientnet-b2', etc.
        num_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Sequential(
            nn.Linear(num_features, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)


def save_model(path, model):
 torch.save(model.state_dict(), path)
 print("Model saved at", path)

def load_model(path, model):
  model.load_state_dict(torch.load(path))
  print("Loaded model from", path)
  return model
