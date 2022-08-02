import torch
from torch import nn
from torchvision.io import read_image
import torchvision.transforms.functional as tvf
from dcnn_srcnn import SRCNN
from dcnn_vdsr import VDSR_Reference

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





        
class DSDMSR_x8(nn.Module):
    def __init__(self, device="cpu"):
        super(DSDMSR_x8, self).__init__()
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

        self.msf_dcnn = SRCNN(num_channels=1)

    def get_sparse_image_batch(self, image_batch, position):
        """Takes an image batch and creates a sparse matrix in the following manner:
            P0: [[P0, 0]  P1: [[0, P1]  P2: [[0, 0]  P3: [[0, 0],
                [0, 0]],       [0, 0]],    [P2, 0]],    [0, P3]].

            Args:
                image_batch (torch.Tensor): A batch of image tensors
                position (int): An integer in [0, 1, 2, 3] -> [00, 01, 10, 11], 
                which tells the position of the pixel in the sparse matrix of 2x2.

            Returns:
                torch.Tensor: A batch of image tensors with the sparse matrices.
        """
        batch_size = image_batch.shape[0]
        n_channels = image_batch.shape[1]
        width, height = image_batch.shape[2], image_batch.shape[3]
        offset_x, offset_y = 0, 0
        if position == 1:
            offset_x = 1
        elif position == 2:
            offset_y = 1
        elif position == 3:
            offset_x = offset_y = 1
        indices = [[((j//height)*2) + offset_y, ((j%height)*2) + offset_x] for j in range(width * height)]
        indices = torch.LongTensor(indices).to(self.device)
        # There are 3 channels in the input image. Every row is a flattened channel after this operation.
        flattened_batch = torch.flatten(image_batch, start_dim=2, end_dim=3)
        dest_size =  torch.Size([width*2, height*2])
        result_tensor = torch.rand(batch_size, n_channels, width*2, height*2).to(self.device)
        for batch_idx in range(batch_size):
            for channel_idx in range(n_channels):
                temp_tensor = torch.sparse_coo_tensor(
                    indices.t(), 
                    flattened_batch[batch_idx][channel_idx], 
                    dest_size).to_dense()
                result_tensor[batch_idx][channel_idx] = temp_tensor.to(self.device)
        del image_batch
        del indices
        del dest_size
        return result_tensor

    
    def novel_view_synthesis_batch(self, image_1_1, image_1_2, image_2_1, image_2_2):
        """Creates a novel view synthesis in the following way:
        P0 <- A pixel from image image_1_1, P1 <- A pixel from image image_1_2, 
        P2 <- A pixel from image image_2_1, P3 <- A pixel from image image_2_2
        Arranges them in the following way: 
        [[P0, P1], 
        [P2, P3]]

        Args:
            image_1_1 (torch.Tensor): A batch of images, which should take position P0.
            image_1_2 (torch.Tensor): A batch of images, which should take position P1.
            image_2_1 (torch.Tensor): A batch of images, which should take position P2.
            image_2_2 (torch.Tensor): A batch of images, which should take position P3.

        Returns:
            torch.Tensor: A novel view synthesis image batch which is twice the input image size.
        """
        image_1_1 = self.get_sparse_image_batch(image_1_1, 0)
        image_1_2 = self.get_sparse_image_batch(image_1_2, 1)
        image_2_1 = self.get_sparse_image_batch(image_2_1, 2)
        image_2_2 = self.get_sparse_image_batch(image_2_2, 3)

        image_temp_1 = torch.add(image_1_1, image_1_2)
        image_temp_2 = torch.add(image_2_1, image_2_2)
        return torch.add(image_temp_1, image_temp_2)
    
    
    def upscale_batch(self, image_batch, scale_factor):
        batch_size, channels, height, width = image_batch.shape
        upscaled_batch = torch.zeros([batch_size, channels, height*scale_factor, width*scale_factor], dtype=torch.float).to(self.device)
        for batch_idx in range(batch_size):
            image = tvf.resize(image_batch[batch_idx], (height*scale_factor, width*scale_factor))
            upscaled_batch[batch_idx, :, :, :] = image
        return upscaled_batch

    
    def forward(self, image):
        out_1_1_x2 = self.dcnn_unit_1_1_x2(image)
        out_1_2_x2 = self.dcnn_unit_1_2_x2(image)
        out_2_1_x2 = self.dcnn_unit_2_1_x2(image)
        out_2_2_x2 = self.dcnn_unit_2_2_x2(image)

        out_x2 = self.novel_view_synthesis_batch(out_1_1_x2, out_1_2_x2, out_2_1_x2, out_2_2_x2)
        
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
        
        out_x2_to_x8 = self.upscale_batch(out_x2, 4)
        out_x4_to_x8 = self.upscale_batch(out_x4, 2)
        
        msf_in_1 = torch.add(out_x2_to_x8, out_x4_to_x8)
        msf_in = torch.add(msf_in_1, out_x8)
        msf_out = self.msf_dcnn(msf_in)
        
        return out_x2, out_x4, out_x8, msf_out



class DSDMSR_VDSR_x8(nn.Module):
    def __init__(self, device="cpu"):
        super(DSDMSR_VDSR_x8, self).__init__()
        self.device = device

        self.dcnn_unit_1_1_x2 = VDSR_Reference(num_channels=1)
        self.dcnn_unit_1_2_x2 = VDSR_Reference(num_channels=1)
        self.dcnn_unit_2_1_x2 = VDSR_Reference(num_channels=1)
        self.dcnn_unit_2_2_x2 = VDSR_Reference(num_channels=1)

        self.dcnn_unit_1_1_x4 = VDSR_Reference(num_channels=1)
        self.dcnn_unit_1_2_x4 = VDSR_Reference(num_channels=1)
        self.dcnn_unit_2_1_x4 = VDSR_Reference(num_channels=1)
        self.dcnn_unit_2_2_x4 = VDSR_Reference(num_channels=1)

        self.dcnn_unit_1_1_x8 = VDSR_Reference(num_channels=1)
        self.dcnn_unit_1_2_x8 = VDSR_Reference(num_channels=1)
        self.dcnn_unit_2_1_x8 = VDSR_Reference(num_channels=1)
        self.dcnn_unit_2_2_x8 = VDSR_Reference(num_channels=1)

        self.msf_dcnn = VDSR_Reference(num_channels=1)

    def get_sparse_image_batch(self, image_batch, position):
        """Takes an image batch and creates a sparse matrix in the following manner:
            P0: [[P0, 0]  P1: [[0, P1]  P2: [[0, 0]  P3: [[0, 0],
                [0, 0]],       [0, 0]],    [P2, 0]],    [0, P3]].

            Args:
                image_batch (torch.Tensor): A batch of image tensors
                position (int): An integer in [0, 1, 2, 3] -> [00, 01, 10, 11], 
                which tells the position of the pixel in the sparse matrix of 2x2.

            Returns:
                torch.Tensor: A batch of image tensors with the sparse matrices.
        """
        batch_size = image_batch.shape[0]
        n_channels = image_batch.shape[1]
        width, height = image_batch.shape[2], image_batch.shape[3]
        offset_x, offset_y = 0, 0
        if position == 1:
            offset_x = 1
        elif position == 2:
            offset_y = 1
        elif position == 3:
            offset_x = offset_y = 1
        indices = [[((j//height)*2) + offset_y, ((j%height)*2) + offset_x] for j in range(width * height)]
        indices = torch.LongTensor(indices).to(self.device)
        # There are 3 channels in the input image. Every row is a flattened channel after this operation.
        flattened_batch = torch.flatten(image_batch, start_dim=2, end_dim=3)
        dest_size =  torch.Size([width*2, height*2])
        result_tensor = torch.rand(batch_size, n_channels, width*2, height*2).to(self.device)
        for batch_idx in range(batch_size):
            for channel_idx in range(n_channels):
                temp_tensor = torch.sparse_coo_tensor(
                    indices.t(), 
                    flattened_batch[batch_idx][channel_idx], 
                    dest_size).to_dense()
                result_tensor[batch_idx][channel_idx] = temp_tensor.to(self.device)
        del image_batch
        del indices
        del dest_size
        return result_tensor

    
    def novel_view_synthesis_batch(self, image_1_1, image_1_2, image_2_1, image_2_2):
        """Creates a novel view synthesis in the following way:
        P0 <- A pixel from image image_1_1, P1 <- A pixel from image image_1_2, 
        P2 <- A pixel from image image_2_1, P3 <- A pixel from image image_2_2
        Arranges them in the following way: 
        [[P0, P1], 
        [P2, P3]]

        Args:
            image_1_1 (torch.Tensor): A batch of images, which should take position P0.
            image_1_2 (torch.Tensor): A batch of images, which should take position P1.
            image_2_1 (torch.Tensor): A batch of images, which should take position P2.
            image_2_2 (torch.Tensor): A batch of images, which should take position P3.

        Returns:
            torch.Tensor: A novel view synthesis image batch which is twice the input image size.
        """
        image_1_1 = self.get_sparse_image_batch(image_1_1, 0)
        image_1_2 = self.get_sparse_image_batch(image_1_2, 1)
        image_2_1 = self.get_sparse_image_batch(image_2_1, 2)
        image_2_2 = self.get_sparse_image_batch(image_2_2, 3)

        image_temp_1 = torch.add(image_1_1, image_1_2)
        image_temp_2 = torch.add(image_2_1, image_2_2)
        return torch.add(image_temp_1, image_temp_2)
    
    
    def upscale_batch(self, image_batch, scale_factor):
        batch_size, channels, height, width = image_batch.shape
        upscaled_batch = torch.zeros([batch_size, channels, height*scale_factor, width*scale_factor], dtype=torch.float).to(self.device)
        for batch_idx in range(batch_size):
            image = tvf.resize(image_batch[batch_idx], (height*scale_factor, width*scale_factor))
            upscaled_batch[batch_idx, :, :, :] = image
        return upscaled_batch

    
    def forward(self, image):
        out_1_1_x2 = self.dcnn_unit_1_1_x2(image)
        out_1_2_x2 = self.dcnn_unit_1_2_x2(image)
        out_2_1_x2 = self.dcnn_unit_2_1_x2(image)
        out_2_2_x2 = self.dcnn_unit_2_2_x2(image)

        out_x2 = self.novel_view_synthesis_batch(out_1_1_x2, out_1_2_x2, out_2_1_x2, out_2_2_x2)
        
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
        
        out_x2_to_x8 = self.upscale_batch(out_x2, 4)
        out_x4_to_x8 = self.upscale_batch(out_x4, 2)
        
        msf_in_1 = torch.add(out_x2_to_x8, out_x4_to_x8)
        msf_in = torch.add(msf_in_1, out_x8)
        msf_out = self.msf_dcnn(msf_in)
        
        return out_x2, out_x4, out_x8, msf_out
        

if __name__ == "__main__":
    # Checking if the model works as intended
    # Let's test the network
    start_time = time.time()
    device = torch.device("cuda:2")
    net = DSDMSR_VDSR_x8(device).to(device)
    print("-"*40, "NETWORK ARCHITECTURE", "-"*40)
    print(net)
    print("-"*105)
    t = (torch.cuda.get_device_properties(0).total_memory) / (1024 * 1024 * 1024)
    r = (torch.cuda.memory_reserved(0)) / (1024 * 1024 * 1024)
    a = (torch.cuda.memory_allocated(0)) / (1024 * 1024 * 1024)
    f = r-a  # free inside reserved
    print(f"Total: {t}G, Reserved: {r}G, Allocated: {a}G, Free: {f}G")
    print("Feeding test input to the model: 1 image of size 80x64")
    test_input = torch.randn((1, 1, 80, 64)).to(device)
    out_x2, out_x4, out_x8, msf_out = net(test_input)
    print(f"Output shape {msf_out.shape}")
    end_time = time.time()
    print(f"Test time: {end_time - start_time}")
