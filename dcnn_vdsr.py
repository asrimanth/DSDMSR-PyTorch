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
    def __init__(self, in_channels=3 , out_channels=3, depth=10):
        """Very Deep Super Resolution (VDSR Net) without residual learning (V1).

        Args:
            in_channels (int, optional): Number of in channels. Defaults to 3.
            out_channels (int, optional): Number of out channels. Defaults to 3.
            depth (int, optional): The number of Conv+ReLU blocks chained together. Defaults to 20.
        """
        super(VDSR_Net_v1, self).__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, padding_mode="zeros"),
            nn.ReLU(True),
        )
        
        depth -= 2
        # self.vdsr_blocks = nn.ModuleList([Conv2dReLU(64, 64, ((2*d)+1), d) for d in range(1, depth+1)])
        self.vdsr_blocks = nn.ModuleList([Conv2dReLU(64, 64, 3, 1) for d in range(depth)])
        self.vdsr_blocks = nn.Sequential(*self.vdsr_blocks) # Unpack all of them into nn.Squential
        self.output_block = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="zeros")
        self.initialize_weights()
    
    
    def forward(self, lr_image):
        lr_image = self.input_block(lr_image)
        lr_image = self.vdsr_blocks(lr_image)
        lr_image = self.output_block(lr_image)
        return lr_image
    
    # Xavier weight Initialization
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # We are using RELU as an activation function.
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))


class VDSR_Net_v2(nn.Module):
    def __init__(self, in_channels=3 , out_channels=3, depth=10):
        """Very Deep Super Resolution (VDSR Net) with residual learning (V2).

        Args:
            in_channels (int, optional): Number of in channels. Defaults to 3.
            out_channels (int, optional): Number of out channels. Defaults to 3.
            depth (int, optional): The number of Conv+ReLU blocks chained together. Defaults to 20.
        """
        super(VDSR_Net_v2, self).__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, padding_mode="zeros"),
            nn.ReLU(True),
        )
        
        depth -= 2
        self.vdsr_blocks = nn.ModuleList([Conv2dReLU(64, 64, 3, 1) for d in range(depth)])
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
                # We are using RELU as an activation function.
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))


class VDSR_Net_v3(nn.Module):
    """
        Very Deep Super Resolution (VDSR Net) with residual learning 
            and depth-based increase in receptive field (V3).

        Args:
            in_channels (int, optional): Number of in channels. Defaults to 3.
            out_channels (int, optional): Number of out channels. Defaults to 3.
            depth (int, optional): The number of Conv+ReLU blocks chained together. Defaults to 20.
    """
    def __init__(self, in_channels=3 , out_channels=3, depth=20):
        super(VDSR_Net_v3, self).__init__()
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
                # We are using RELU as an activation function.
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))



# Architecture for reference. Taken from:
# https://github.com/Lornatang/VDSR-PyTorch/blob/master/model.py
from math import sqrt

class ConvReLU(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)

        return out

class VDSR_Reference(nn.Module):
    def __init__(self, num_channels=1) -> None:
        super(VDSR_Reference, self).__init__()
        # Input layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.ReLU(True),
        )

        # Features trunk blocks
        trunk = []
        for _ in range(8):
            trunk.append(ConvReLU(64))
        self.trunk = nn.Sequential(*trunk)

        # Output layer
        self.conv2 = nn.Conv2d(64, num_channels, (3, 3), (1, 1), (1, 1), bias=False)

        # Initialize model weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.trunk(out)
        out = self.conv2(out)

        out = torch.add(out, identity)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0.0, sqrt(2 / (module.kernel_size[0] * module.kernel_size[1] * module.out_channels)))


if __name__ == "__main__":
    # Checking if the model works as intended
    # Let's test the VDSR network
    vdsr_net = VDSR_Reference()
    print("-"*40, "NETWORK ARCHITECTURE", "-"*40)
    print(vdsr_net)
    print("-"*105)
    print("Feeding test input to the model: 8 images of size 240x240")
    test_input = torch.randn((8, 1, 240, 240))
    out = vdsr_net(test_input)
    print(f"Output shape {out.shape}")
