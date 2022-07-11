import torch
from torch import nn

class Conv2dReLU(nn.Module):
    """A block which performs 2D convolution and ReLU sequentially.

    Args:
        in_chnl (int): Number of in channels.
        out_chnl (int):  Number of out channels.
        kernel_size (int or tuple): Size of the kernel (otherwise known as receptive field).
        padding (int): The number of pixels to add at the edges. By default, black pixels are added.
    """
    def __init__(self, in_chnl, out_chnl, kernel_size, padding):
        super(Conv2dReLU, self).__init__()
        self.conv = nn.Conv2d(in_chnl, out_chnl, 
                              kernel_size=kernel_size, 
                              stride=1, 
                              padding=padding, 
                              padding_mode="zeros")
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, data):
        data = self.conv(data)
        data = self.relu(data)
        return data




class VDSR_Net_v1(nn.Module):
    """Very Deep Super Resolution (VDSR Net) with residual learning 
            and depth-based increase in receptive field (V3).

        Args:
            in_channels (int, optional): Number of in channels. Defaults to 3.
            out_channels (int, optional): Number of out channels. Defaults to 3.
            depth (int, optional): The number of Conv+ReLU blocks chained together. Defaults to 10.
        """
    def __init__(self, in_channels=3 , out_channels=3, depth=10):
        super(VDSR_Net_v1, self).__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, padding_mode="zeros"),
            nn.ReLU(True),
        )
        
        depth -= 2
        self.vdsr_blocks = nn.ModuleList([Conv2dReLU(64, 64, 3, 1) for d in range(1, depth+1)])
        self.vdsr_blocks = nn.Sequential(*self.vdsr_blocks) # Unpack all of them into nn.Squential
        self.output_block = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="zeros")
        # Weight initialization
        self.initialize_weights()
    
    
    def forward(self, lr_image):
        original = lr_image.clone()
        lr_image = self.input_block(lr_image)
        lr_image = self.vdsr_blocks(lr_image)
        lr_image = self.output_block(lr_image)
        # Residual learning
        lr_image = torch.add(lr_image, original)
        return lr_image
    
    # Xavier weight Initialization
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1)



class VDSR_Net_v2(nn.Module):
    """Very Deep Super Resolution (VDSR Net) with residual learning 
            and depth-based increase in receptive field (V3).

        Args:
            in_channels (int, optional): Number of in channels. Defaults to 3.
            out_channels (int, optional): Number of out channels. Defaults to 3.
            depth (int, optional): The number of Conv+ReLU blocks chained together. Defaults to 10.
        """
    def __init__(self, in_channels=3 , out_channels=3, depth=10):
        super(VDSR_Net_v2, self).__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, padding_mode="zeros"),
            nn.ReLU(True),
        )
        
        depth -= 2
        self.vdsr_blocks = nn.ModuleList([Conv2dReLU(64, 64, ((2*d)+3), d+1) for d in range(1, depth+1)])
        self.vdsr_blocks = nn.Sequential(*self.vdsr_blocks) # Unpack all of them into nn.Squential
        self.output_block = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="zeros")
        # Weight initialization
        self.initialize_weights()
    
    
    def forward(self, lr_image):
        original = lr_image.clone()
        lr_image = self.input_block(lr_image)
        lr_image = self.vdsr_blocks(lr_image)
        lr_image = self.output_block(lr_image)
        # Residual learning
        lr_image = torch.add(lr_image, original)
        return lr_image
    
    # Xavier weight Initialization
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1)

                


if __name__ == "__main__":
    # Checking if the model works as intended
    # Let's test the VDSR network
    vdsr_net = VDSR_Net_v1()
    print("-"*40, "NETWORK ARCHITECTURE", "-"*40)
    print(vdsr_net)
    print("-"*105)
    print("Feeding test input to the model: A single grayscale image of size 128x128")
    test_input = torch.randn((1, 3, 41, 41))
    out = vdsr_net(test_input)
    print(f"Output shape {out.shape}")