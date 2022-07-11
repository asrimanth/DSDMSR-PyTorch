import torch
from torch import nn
from torchvision.io import read_image
import torchvision.transforms.functional as tvf
from dcnn_srcnn import *

import time

class DSDMSR(nn.Module):
    def __init__(self, device="cuda"):
        super(DSDMSR, self).__init__()
        self.device = device
        
        self.dcnn_unit_1_1_x2 = SRCNN(num_channels=1)
        self.dcnn_unit_1_2_x2 = SRCNN(num_channels=1)
        self.dcnn_unit_2_1_x2 = SRCNN(num_channels=1)
        self.dcnn_unit_2_2_x2 = SRCNN(num_channels=1)
        
        self.dcnn_unit_1_1_x4 = SRCNN(num_channels=1)
        self.dcnn_unit_1_2_x4 = SRCNN(num_channels=1)
        self.dcnn_unit_2_1_x4 = SRCNN(num_channels=1)
        self.dcnn_unit_2_2_x4 = SRCNN(num_channels=1)
        
        self.dcnn_unit_1_1_x8 = SRCNN(num_channels=1)
        self.dcnn_unit_1_2_x8 = SRCNN(num_channels=1)
        self.dcnn_unit_2_1_x8 = SRCNN(num_channels=1)
        self.dcnn_unit_2_2_x8 = SRCNN(num_channels=1)
        
        self.dcnn_unit_1_1_x16 = SRCNN(num_channels=1)
        self.dcnn_unit_1_2_x16 = SRCNN(num_channels=1)
        self.dcnn_unit_2_1_x16 = SRCNN(num_channels=1)
        self.dcnn_unit_2_2_x16 = SRCNN(num_channels=1)
        
        self.msf_dcnn = SRCNN(num_channels=1)
    
    def novel_view_synthesis(self, image_1_1, image_1_2, image_2_1, image_2_2):
        channels, h, w = image_1_1.shape
        canvas = torch.zeros([channels, h*2, w*2], dtype=torch.float).to(self.device)
        even_images = [image_1_1, image_1_2]
        odd_images = [image_2_1, image_2_2]
        channels, height, width = canvas.shape
        for h in range(height):
            if h % 2 == 0:
                images = even_images
            else:
                images = odd_images
            for w in range(width):
                if w % 2 == 0:
                    image_to_place = images[0]
                else:
                    image_to_place = images[1]
                canvas[:, h, w] = image_to_place[:, h//2, w//2]
        del images
        del image_to_place
        del even_images
        del odd_images
        return canvas
    
    def novel_view_synthesis_batch(self, image_1_1_batch, image_1_2_batch, image_2_1_batch, image_2_2_batch):
        batch_size, channels, height, width = image_1_1_batch.shape
        canvas_batch = torch.zeros([batch_size, channels, height*2, width*2], dtype=torch.float).to(self.device)
        for batch_idx in range(batch_size):
            image_1_1 = image_1_1_batch[batch_idx]
            image_1_2 = image_1_2_batch[batch_idx]
            image_2_1 = image_2_1_batch[batch_idx]
            image_2_2 = image_2_2_batch[batch_idx]

            even_images = [image_1_1, image_1_2]
            odd_images = [image_2_1, image_2_2]
            canvas = self.novel_view_synthesis(image_1_1, image_1_2, image_2_1, image_2_2)
            canvas_batch[batch_idx, :, :, :] = canvas
        return canvas_batch
    
    def upscale(self, image_batch, scale_factor):
        batch_size, channels, height, width = image_batch.shape
        upscaled_batch = torch.zeros([batch_size, channels, height*scale_factor, width*scale_factor], dtype=torch.float).to(self.device)
        for batch_idx in range(batch_size):
            image = tvf.resize(image_batch[batch_idx], (height*scale_factor, width*scale_factor))
            upscaled_batch[batch_idx, :, :, :] = image
        return upscaled_batch

    def forward(self, image):
        # print(f"Type of image: {image.type()}")
        out_1_1_x2 = self.dcnn_unit_1_1_x2(image)
        out_1_2_x2 = self.dcnn_unit_1_2_x2(image)
        out_2_1_x2 = self.dcnn_unit_2_1_x2(image)
        out_2_2_x2 = self.dcnn_unit_2_2_x2(image)
        out_x2 = self.novel_view_synthesis_batch(out_1_1_x2, out_1_2_x2, out_2_1_x2, out_2_2_x2)
        # print(f"Type of out_x2: {out_x2.type()}")
        
        out_1_1_x4 = self.dcnn_unit_1_1_x4(out_x2)
        out_1_2_x4 = self.dcnn_unit_1_2_x4(out_x2)
        out_2_1_x4 = self.dcnn_unit_2_1_x4(out_x2)
        out_2_2_x4 = self.dcnn_unit_2_2_x4(out_x2)
        out_x4 = self.novel_view_synthesis_batch(out_1_1_x4, out_1_2_x4, out_2_1_x4, out_2_2_x4)
        
        out_1_1_x8 = self.dcnn_unit_1_1_x8(out_x4)
        out_1_2_x8 = self.dcnn_unit_1_2_x8(out_x4)
        out_2_1_x8 = self.dcnn_unit_2_1_x8(out_x4)
        out_2_2_x8 = self.dcnn_unit_2_2_x8(out_x4)
        out_x8 = self.novel_view_synthesis_batch(out_1_1_x8, out_1_2_x8, out_2_1_x8, out_2_2_x8)
        
        out_1_1_x16 = self.dcnn_unit_1_1_x16(out_x8)
        out_1_2_x16 = self.dcnn_unit_1_2_x16(out_x8)
        out_2_1_x16 = self.dcnn_unit_2_1_x16(out_x8)
        out_2_2_x16 = self.dcnn_unit_2_2_x16(out_x8)
        out_x16 = self.novel_view_synthesis_batch(out_1_1_x16, out_1_2_x16, out_2_1_x16, out_2_2_x16)
        
        out_x2_to_x16 = self.upscale(out_x2, 8)
        out_x4_to_x16 = self.upscale(out_x4, 4)
        out_x8_to_x16 = self.upscale(out_x8, 2)
        
        msf_in_1 = torch.add(out_x2_to_x16, out_x4_to_x16)
        msf_in_2 = torch.add(out_x8_to_x16, out_x16)
        msf_in = torch.add(msf_in_1, msf_in_2)
        msf_out = self.msf_dcnn(msf_in)
        
        return out_x2, out_x4, out_x8, out_x16, msf_out
        

if __name__ == "__main__":
    # Checking if the model works as intended
    # Let's test the network
    start_time = time.time()
    net = DSDMSR().to("cuda")
    print("-"*40, "NETWORK ARCHITECTURE", "-"*40)
    print(net)
    print("-"*105)
    print("Feeding test input to the model: 8 images of size 32x32")
    test_input = torch.randn((8, 1, 32, 32)).to("cuda")
    out_x2, out_x4, out_x8, out_x16, msf_out = net(test_input)
    print(f"Output shape {msf_out.shape}")
    end_time = time.time()
    print(f"Test time: {end_time - start_time}")
