import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

if __name__ == "__main__":
    # Checking if the model works as intended
    # Let's test the network
    net = SRCNN(3)
    print("-"*40, "NETWORK ARCHITECTURE", "-"*40)
    print(net)
    print("-"*105)
    print("Feeding test input to the model: A single image of size 128x128")
    test_input = torch.randn((1, 3, 128, 128))
    out = net(test_input)
    print(f"Output shape {out.shape}")