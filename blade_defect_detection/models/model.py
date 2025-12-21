"""UNet model for blade defect segmentation."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Double convolution block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights to prevent NaN values."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Simple UNet architecture for segmentation."""

    def __init__(self, in_channels: int = 3, num_classes: int = 5):
        """Initialize UNet.

        Args:
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes (background + 4 defects = 5)
        """
        super().__init__()
        self.num_classes = num_classes

        self.enc1 = DoubleConv(in_channels, 40)
        self.enc2 = DoubleConv(40, 80)
        self.enc3 = DoubleConv(80, 160)
        self.enc4 = DoubleConv(160, 320)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(320, 640)
        self.up4 = nn.ConvTranspose2d(640, 320, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(640, 320)

        self.up3 = nn.ConvTranspose2d(320, 160, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(320, 160)

        self.up2 = nn.ConvTranspose2d(160, 80, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(160, 80)

        self.up1 = nn.ConvTranspose2d(80, 40, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(80, 40)

        # Final output layer
        self.final = nn.Conv2d(40, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Output tensor [B, num_classes, H, W]
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)

        return out


class BladeDefectModel(nn.Module):
    """Wrapper for UNet model."""

    def __init__(self, in_channels: int = 3, num_classes: int = 5):
        """Initialize model.

        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
        """
        super().__init__()
        self.model = UNet(in_channels=in_channels, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

