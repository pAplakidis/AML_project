import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


# code from [ https://github.com/vinceecws/SegNet_PyTorch ]
class ComboSegModel(nn.Module):
  def __init__(self, ch_in, ch_out, BN_momentum=0.5):
    super(ComboSegModel, self).__init__()

    self.in_chn= ch_in
    self.out_chn = ch_out

    self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True)

    # Encoder
    self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
    self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
    self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)

    self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
    self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

    self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
    self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
    self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)

    self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
    self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
    self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
    self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

    self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
    self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
    self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)

    # Decoder
    self.MaxDe = nn.MaxUnpool2d(2, stride=2) 

    self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.BNDe53 = nn.BatchNorm2d(512, momentum=BN_momentum)
    self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.BNDe52 = nn.BatchNorm2d(512, momentum=BN_momentum)
    self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.BNDe51 = nn.BatchNorm2d(512, momentum=BN_momentum)

    self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.BNDe43 = nn.BatchNorm2d(512, momentum=BN_momentum)
    self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.BNDe42 = nn.BatchNorm2d(512, momentum=BN_momentum)
    self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
    self.BNDe41 = nn.BatchNorm2d(256, momentum=BN_momentum)

    self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    self.BNDe33 = nn.BatchNorm2d(256, momentum=BN_momentum)
    self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    self.BNDe32 = nn.BatchNorm2d(256, momentum=BN_momentum)
    self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
    self.BNDe31 = nn.BatchNorm2d(128, momentum=BN_momentum)

    self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.BNDe22 = nn.BatchNorm2d(128, momentum=BN_momentum)
    self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
    self.BNDe21 = nn.BatchNorm2d(64, momentum=BN_momentum)

    self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    self.BNDe12 = nn.BatchNorm2d(64, momentum=BN_momentum)
    self.ConvDe11 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)
    self.BNDe11 = nn.BatchNorm2d(self.out_chn, momentum=BN_momentum)
    
     # Classification head
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(512 * 4 * 4, 1024) # TODO: Check if 2 are necessary 
    self.fc2 = nn.Linear(1024, 4) 

  def forward(self, x):
    #ENCODE LAYERS
    #Stage 1
    x = F.relu(self.BNEn11(self.ConvEn11(x)))
    x = F.relu(self.BNEn12(self.ConvEn12(x)))
    x, ind1 = self.MaxEn(x)
    size1 = x.size()

    #Stage 2
    x = F.relu(self.BNEn21(self.ConvEn21(x))) 
    x = F.relu(self.BNEn22(self.ConvEn22(x))) 
    x, ind2 = self.MaxEn(x)
    size2 = x.size()

    #Stage 3
    x = F.relu(self.BNEn31(self.ConvEn31(x))) 
    x = F.relu(self.BNEn32(self.ConvEn32(x))) 
    x = F.relu(self.BNEn33(self.ConvEn33(x)))   
    x, ind3 = self.MaxEn(x)
    size3 = x.size()

    #Stage 4
    x = F.relu(self.BNEn41(self.ConvEn41(x))) 
    x = F.relu(self.BNEn42(self.ConvEn42(x))) 
    x = F.relu(self.BNEn43(self.ConvEn43(x)))   
    x, ind4 = self.MaxEn(x)
    size4 = x.size()

    #Stage 5
    x = F.relu(self.BNEn51(self.ConvEn51(x))) 
    x = F.relu(self.BNEn52(self.ConvEn52(x))) 
    x = F.relu(self.BNEn53(self.ConvEn53(x)))   
    x, ind5 = self.MaxEn(x)
    size5 = x.size()

    # print(f'Latent space shape: {x.shape}') Latent space shape: torch.Size([32, 512, 4, 4])

    # Classification head
    latent_space = x
    cls_output = self.flatten(latent_space)
    cls_output = F.relu(self.fc1(cls_output)) 
    cls_output = self.fc2(cls_output)

    #DECODE LAYERS
    #Stage 5
    x = self.MaxDe(x, ind5, output_size=size4)
    x = F.relu(self.BNDe53(self.ConvDe53(x)))
    x = F.relu(self.BNDe52(self.ConvDe52(x)))
    x = F.relu(self.BNDe51(self.ConvDe51(x)))

    #Stage 4
    x = self.MaxDe(x, ind4, output_size=size3)
    x = F.relu(self.BNDe43(self.ConvDe43(x)))
    x = F.relu(self.BNDe42(self.ConvDe42(x)))
    x = F.relu(self.BNDe41(self.ConvDe41(x)))

    #Stage 3
    x = self.MaxDe(x, ind3, output_size=size2)
    x = F.relu(self.BNDe33(self.ConvDe33(x)))
    x = F.relu(self.BNDe32(self.ConvDe32(x)))
    x = F.relu(self.BNDe31(self.ConvDe31(x)))

    #Stage 2
    x = self.MaxDe(x, ind2, output_size=size1)
    x = F.relu(self.BNDe22(self.ConvDe22(x)))
    x = F.relu(self.BNDe21(self.ConvDe21(x)))

    #Stage 1
    x = self.MaxDe(x, ind1)
    x = F.relu(self.BNDe12(self.ConvDe12(x)))
    x = self.ConvDe11(x)

    x = F.softmax(x, dim=1)

    return x, cls_output

class ComboUnetModel(nn.Module):
  def __init__(self, in_channels=3, out_channels=1, init_features=32, BN_momentum=0.5):
    super(ComboUnetModel, self).__init__()

    features = init_features
    self.encoder1 = self._block(in_channels, features, name="enc1")
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.encoder2 = self._block(features, features * 2, name="enc2")
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.encoder3 = self._block(features * 2, features * 4, name="enc3")
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.encoder4 = self._block(features * 4, features * 8, name="enc4")
    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

    self.upconv4 = nn.ConvTranspose2d(
        features * 16, features * 8, kernel_size=2, stride=2
    )
    self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
    self.upconv3 = nn.ConvTranspose2d(
        features * 8, features * 4, kernel_size=2, stride=2
    )
    self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
    self.upconv2 = nn.ConvTranspose2d(
        features * 4, features * 2, kernel_size=2, stride=2
    )
    self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
    self.upconv1 = nn.ConvTranspose2d(
        features * 2, features, kernel_size=2, stride=2
    )
    self.decoder1 = self._block(features * 2, features, name="dec1")

    self.conv = nn.Conv2d(
        in_channels=features, out_channels=out_channels, kernel_size=1
    )

    # Classification head
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(32768, 1024) # TODO: Check if 2 are necessary 
    self.fc2 = nn.Linear(1024, 4) 

  def forward(self, x):
    # ENCODER
    enc1 = self.encoder1(x)
    enc2 = self.encoder2(self.pool1(enc1))
    enc3 = self.encoder3(self.pool2(enc2))
    enc4 = self.encoder4(self.pool3(enc3))
    bottleneck = self.bottleneck(self.pool4(enc4))

    # Classification head
    latent_space = bottleneck
    cls_output = self.flatten(latent_space)
    cls_output = F.relu(self.fc1(cls_output)) 
    cls_output = self.fc2(cls_output)

    # DECODER
    dec4 = self.upconv4(bottleneck)
    dec4 = torch.cat((dec4, enc4), dim=1)
    dec4 = self.decoder4(dec4)
    dec3 = self.upconv3(dec4)
    dec3 = torch.cat((dec3, enc3), dim=1)
    dec3 = self.decoder3(dec3)
    dec2 = self.upconv2(dec3)
    dec2 = torch.cat((dec2, enc2), dim=1)
    dec2 = self.decoder2(dec2)
    dec1 = self.upconv1(dec2)
    dec1 = torch.cat((dec1, enc1), dim=1)
    dec1 = self.decoder1(dec1)
    seg_out = self.conv(dec1)

    # seg_out = torch.sigmoid(seg_out)
    return seg_out, cls_output

  @staticmethod
  def _block(in_channels, features, name):
      return nn.Sequential(
          OrderedDict(
              [
                  (
                      name + "conv1",
                      nn.Conv2d(
                          in_channels=in_channels,
                          out_channels=features,
                          kernel_size=3,
                          padding=1,
                          bias=False,
                      ),
                  ),
                  (name + "norm1", nn.BatchNorm2d(num_features=features)),
                  (name + "relu1", nn.ReLU(inplace=True)),
                  (
                      name + "conv2",
                      nn.Conv2d(
                          in_channels=features,
                          out_channels=features,
                          kernel_size=3,
                          padding=1,
                          bias=False,
                      ),
                  ),
                  (name + "norm2", nn.BatchNorm2d(num_features=features)),
                  (name + "relu2", nn.ReLU(inplace=True)),
              ]
          )
      )



def save_model(path, model):
 torch.save(model.state_dict(), path)
 print("Model saved at", path)

def load_model(path, model):
  model.load_state_dict(torch.load(path))
  print("Loaded model from", path)
  return model
