#!/usr/bin/env python3
import random

from dataset import *
from model import *
from utils import *

from dataset import MultitaskDataset

N_SAMPLES = 5

MODEL_PATH = os.getenv("MODEL_PATH")
if MODEL_PATH == None:
  MODEL_PATH = "models/segnet.pt"
print("[+] Model save path:", MODEL_PATH)


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  # get data
  dataset = MultitaskDataset(BASE_DIR_SEG)
  # dataset = MultitaskDataset(BASE_DIR_SEG_TEST, test=True)

  # define model and train
  in_samp = dataset[0]['image']
  in_ch, out_ch = in_samp.shape[0], 4
  model = ComboSegModel(in_ch, out_ch).to(device)
  model = load_model(MODEL_PATH, model)

  with torch.no_grad():
    model.eval()
    print("[*] Test images preview")
    for i in range(N_SAMPLES):
      samp = dataset[random.randint(0, len(dataset))]
      # img = samp['image']
      img, mask, label = samp['image'], samp['mask'], samp['label']
      out_img = np.moveaxis(img, 0, -1)
      X = torch.tensor([img, img]).float().to(device)
      pred = model(X)
      print(pred)
      # view_net_result(out_img, pred[0])
      view_net_result(out_img, pred[0][0], mask, torch.argmax(pred[1][0]), label, dataset.classes)
