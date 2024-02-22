import torch.nn as nn
import torch

class Resnet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            ResBlock(64, 128, 32, projection=True),
            ResBlock(128, 128, 32),
            ResBlock(128, 128, 32),
        )
        self.conv3 = nn.Sequential(
            ResBlock(128, 256, 64, downsample=True, projection=True),
            ResBlock(256, 256, 64),
            ResBlock(256, 256, 64),
            ResBlock(256, 256, 64),
        )
        self.conv4 = nn.Sequential(
            ResBlock(256, 512, 128, downsample=True, projection=True),
            ResBlock(512, 512, 128),
            ResBlock(512, 512, 128),
            ResBlock(512, 512, 128),
            ResBlock(512, 512, 128),
        )
        self.conv5 = nn.Sequential(
            ResBlock(512, 1024, 256, downsample=True, projection=True),
            ResBlock(1024, 1024, 256),
            ResBlock(1024, 1024, 256),
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(1024, 1000),
            nn.BatchNorm1d(1000),
            nn.GELU(),
            nn.Dropout1d(p=0.5),
            nn.Linear(1000, 10),
        )

    def forward(self, x) -> torch.Tensor:
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
    def __init__(self, in_channels, out_channels, squeeze_channels, downsample=False, projection=False):
        super().__init__()
        if downsample:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(squeeze_channels),
                nn.GELU()
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, squeeze_channels, kernel_size=1),
                nn.BatchNorm2d(squeeze_channels),
                nn.GELU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(squeeze_channels, squeeze_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(squeeze_channels),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(squeeze_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1),
            nn.BatchNorm2d(out_channels),
        ) if projection else nn.Identity()
        self.gelu = nn.GELU()

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        return self.gelu(output + self.identity(x))
    
def main(): 
    model = Resnet50()
    print(sum(p.numel() for p in model.parameters()))
    
if __name__ == "__main__":
    main()