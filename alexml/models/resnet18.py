import torch.nn as nn
import torch

class Resnet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
        )
        self.conv3 = nn.Sequential(
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 128),
        )
        self.conv4 = nn.Sequential(
            ResBlock(128, 256, downsample=True),
            ResBlock(256, 256),
        )
        self.conv5 = nn.Sequential(
            ResBlock(256, 512, downsample=True),
            ResBlock(512, 512),
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.pooling(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2) if downsample else nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2) if downsample else nn.Identity()
        self.gelu = nn.GELU()

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        return self.gelu(output + self.identity(x))
    
def main(): 
    model = Resnet18()
    print(sum(p.numel() for p in model.parameters()))
    
if __name__ == "__main__":
    main()