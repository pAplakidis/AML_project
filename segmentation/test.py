#!/usr/bin/env python3
from dataset import *
from model import *
from utils import *

N_SAMPLES = 5

MODEL_PATH = os.getenv("MODEL_PATH")
if MODEL_PATH == None:
  MODEL_PATH = "models/segnet.pt"
print("[+] Model save path:", MODEL_PATH)


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  # get data
  dataset = SegDataset(BASE_DIR_SEG_TEST, test=True)

  # define model and train
  in_samp = dataset[0]['image']
  in_ch, out_ch = in_samp.shape[0], 4
  model = SegNet(in_ch, out_ch).to(device)
  model = load_model(MODEL_PATH, model)

  classes = np.zeros((4, 3))

  with torch.no_grad():
    model.eval()
    print("[*] Test images preview")
    for i in range(N_SAMPLES):
      samp = dataset[random.randint(0, len(dataset))]
      img = samp['image']
      out_img = np.moveaxis(img, 0, -1)
      X = torch.tensor([img, img]).float().to(device)
      pred = model(X)
      print(pred)
      view_net_result(out_img, pred[0], classes=classes)
