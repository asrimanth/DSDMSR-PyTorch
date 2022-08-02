import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.functional as tvf
import torchvision.transforms as transforms
from PIL import Image

import numpy as np
import pandas as pd


class PatchedDatasetTensor(Dataset):
    def __init__(self, csv_path, image_transforms=None):
        self.data = pd.read_csv(csv_path)
        self.image_transforms = image_transforms
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        x1_image = read_image(self.data.iloc[index, 1])
        x2_image = read_image(self.data.iloc[index, 4])
        x4_image = read_image(self.data.iloc[index, 7])
        x8_image = read_image(self.data.iloc[index, 10])
        x16_image = read_image(self.data.iloc[index, 13])

        x1_image = x1_image.to(torch.float) / 255.0
        x2_image = x2_image.to(torch.float) / 255.0
        x4_image = x4_image.to(torch.float) / 255.0
        x8_image = x8_image.to(torch.float) / 255.0
        x16_image = x16_image.to(torch.float) / 255.0

        return x1_image, x2_image, x4_image, x8_image, x16_image

class Patched_DTU_Tensor(Dataset):
    def __init__(self, csv_path, image_transforms=None):
        self.data = pd.read_csv(csv_path)
        self.image_transforms = image_transforms
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        x64_image = read_image(self.data.iloc[index, 1])
        x128_image = read_image(self.data.iloc[index, 4])
        x256_image = read_image(self.data.iloc[index, 7])
        x512_image = read_image(self.data.iloc[index, 10])
        x1024_image = read_image(self.data.iloc[index, 13])

        x64_image = x64_image.to(torch.float) / 255.0
        x128_image = x128_image.to(torch.float) / 255.0
        x256_image = x256_image.to(torch.float) / 255.0
        x512_image = x512_image.to(torch.float) / 255.0
        x1024_image = x1024_image.to(torch.float) / 255.0
        
        return x64_image, x128_image, x256_image, x512_image, x1024_image


if __name__ == "__main__":
    train_dataset = PatchedDatasetTensor("./train.csv")
    x1, x2, x4, x8, x16 = train_dataset[0]
