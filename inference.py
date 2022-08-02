import torch
import torchvision.transforms as transforms
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from torchvision.io import read_image
import torchvision.transforms.functional as tvf
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from torch.utils.data import DataLoader

from PIL import Image

from model import *
from dataclass import PatchedDatasetTensor, Patched_DTU_Tensor
from utils import *


def get_difference(tensor_image_1, tensor_image_2):
        image_1 = tensor_image_1.cpu().detach().numpy()
        image_2 = tensor_image_2.cpu().detach().numpy()

        difference = image_1 - image_2
        return torch.from_numpy(difference)

# https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
def save_current_prediction(images_list, dest_path):
    widths, heights = zip(*(i.size for i in images_list))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images_list:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save(dest_path)
    new_im.close()


def test_report(model, test_dataloader, device):
    psnr_avg_x2 = 0
    psnr_avg_x4 = 0
    psnr_avg_x8 = 0
    index = 0
    with torch.no_grad():
        for x64_image, x128_image, x256_image, x512_image, _ in tqdm(test_dataloader):
            
            x64_image =  x64_image.to(device)
            x128_image =  x128_image.to(device)
            x256_image =  x256_image.to(device)
            x512_image =  x512_image.to(device)
            
            out_x128, out_x256, out_x512, msf_out = model(x64_image) #.clamp(0.0, 1.0)
            out_x128 = out_x128.clamp(0.0, 1.0)
            out_x256 = out_x256.clamp(0.0, 1.0)
            out_x512 = out_x512.clamp(0.0, 1.0)
            msf_out = msf_out.clamp(0.0, 1.0)

            psnr_avg_x2 += calc_psnr(out_x128.data.cpu(), x128_image.cpu()).item()
            psnr_avg_x4 += calc_psnr(out_x256.data.cpu(), x256_image.cpu()).item()
            psnr_avg_x8 += calc_psnr(out_x512.data.cpu(), x512_image.cpu()).item()

            images_list = [x64_image.squeeze(0),
                           out_x128.squeeze(0), x128_image.squeeze(0),
                           out_x256.squeeze(0), x256_image.squeeze(0),
                           out_x512.squeeze(0), msf_out.squeeze(0), x512_image.squeeze(0)]
            images_list = list(map(tvf.to_pil_image, images_list))
            save_current_prediction(images_list, f"./dtu_results/pred_{index}.png")
            index += 1

    psnr_avg_x2 /= len(test_dataloader)
    psnr_avg_x4 /= len(test_dataloader)
    psnr_avg_x8 /= len(test_dataloader)
    return [("PSNR from the model - x2", psnr_avg_x2), 
            ("PSNR from the model - x4", psnr_avg_x4),
            ("PSNR from the model - x8", psnr_avg_x8)]


if __name__ == "__main__":
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # model = DSDMSR(device)
    # model.load_state_dict(torch.load("./models/best_model_scarlet-frost-11.pth"))
    model = DSDMSR_x8(device)
    model.load_state_dict(torch.load("./models/olive-sponge-25/best_model_olive-sponge-25.pth"))
    model.eval()
    model.to(device)
    
    print(device)
    # print(f"running with device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    test_dataset = Patched_DTU_Tensor("./valid_dtu_sub.csv")
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    # valid_dataset = PatchedDatasetTensor(config.VALID_PATH, test_transforms)
    print(test_report(model, test_dataloader, device))
