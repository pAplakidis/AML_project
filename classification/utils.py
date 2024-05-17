from datetime import datetime

W = H = 128

BASE_DIR_TRAIN = "../data/Brain_Tumor_MRI_Dataset/Training"
BASE_DIR_TEST = "../data/Brain_Tumor_MRI_Dataset/Testing"

BASE_DIR_SEG = "../data/BraTS2020/train"
BASE_DIR_SEG_VAL = "../data/BraTS2020/eval"
BASE_DIR_SEG_TEST = "../data/BraTS2020/test"

TB_PATH = "runs/" + str(datetime.now())
