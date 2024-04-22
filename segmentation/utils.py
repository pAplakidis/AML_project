import numpy as np

W = H = 224
BASE_DIR_SEG = "../data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/input_data_3channels"

# 1-hot encodes a tensor
def to_categorical(y, num_classes):
  return np.eye(num_classes, dtype='uint8')[y]

# TODO: test this (might be ignoring classes)
def from_categorical(encoded_array):
  return np.argmax(encoded_array, axis=-1)
