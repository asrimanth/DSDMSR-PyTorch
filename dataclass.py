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


class TestDataset(Dataset):
    def __init__(self, csv_path, image_transforms=None):
        self.data = pd.read_csv(csv_path)
        self.image_transforms = image_transforms

    def __len__(self):
        return self.data.shape[0]
    
    def get_difference(self, tensor_image_1, tensor_image_2):
        image_1 = tensor_image_1.detach().numpy()
        image_2 = tensor_image_2.detach().numpy()

        difference = image_1 - image_2
        return torch.from_numpy(difference)
    
    def __getitem__(self, index):
        lr_image = Image.open(self.data.iloc[index, 2]).convert("RGB")
        hr_image = Image.open(self.data.iloc[index, 5]).convert("RGB")

        pil_to_tensor = transforms.PILToTensor()
        lr_image = pil_to_tensor(lr_image)
        hr_image = pil_to_tensor(hr_image)
        lr_image = lr_image.to(torch.float)
        hr_image = hr_image.to(torch.float)

        if self.image_transforms is not None:
            sample = {'lr_image': lr_image, 'hr_image': hr_image}
            out_sample = self.image_transforms(sample)
            lr_image, hr_image = out_sample['lr_image'], out_sample['hr_image']
        
        return lr_image, hr_image, self.get_difference(lr_image, hr_image)


if __name__ == "__main__":
    train_dataset = PatchedDatasetTensor("./train.csv")
    x1, x2, x4, x8, x16 = train_dataset[0]
