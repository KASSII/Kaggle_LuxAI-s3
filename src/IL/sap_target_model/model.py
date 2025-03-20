import torch
import torch.nn as nn

class ILModel(torch.nn.Module):
    def __init__(self, in_c, global_feature_ch=11):
        super(ILModel, self).__init__()

        # Contracting path
        self.enc1 = self._conv_block(in_c, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self._conv_block(256, 256)

        # Expanding path
        self.dec3 = self._conv_block(256 + 256 + global_feature_ch, 256)
        self.dec2 = self._conv_block(256 + 128, 128)
        self.dec1 = self._conv_block(128 + 64, 64)

        # Final output layer
        self.last_layer = nn.Conv2d(64, 1, kernel_size=1)

        # Pooling and Upsampling
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _conv_block(self, in_channels, out_channels):
        """Helper method to create a convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, global_features):
        # Contracting path
        x1 = self.enc1(x)  # 24x24 -> 24x24
        p1 = self.pool(x1)  # 24x24 -> 12x12

        x2 = self.enc2(p1)  # 12x12 -> 12x12
        p2 = self.pool(x2)  # 12x12 -> 6x6

        x3 = self.enc3(p2)  # 6x6 -> 6x6
        p3 = self.pool(x3)  # 6x6 -> 3x3

        # Bottleneck
        bottleneck = self.bottleneck(p3)  # 3x3 -> 3x3

        # Global features integration
        global_features = global_features.unsqueeze(-1).unsqueeze(-1)  # (N, 11) -> (N, 11, 1, 1)
        global_features = global_features.expand(-1, -1, 3, 3)  # (N, 11, 3, 3)
        bottleneck = torch.cat([bottleneck, global_features], dim=1)  # (N, 256 + 11, 4, 4)

        # Expanding path
        up3 = self.upsample(bottleneck)  # 3x3 -> 6x6
        up3 = torch.cat([up3, x3], dim=1)  # Concatenate skip connection (N, 256+11+256, 6, 6)
        up3 = self.dec3(up3)

        up2 = self.upsample(up3)  # 6x6 -> 12x12
        up2 = torch.cat([up2, x2], dim=1)  # (N, 256+128, 12, 12)
        up2 = self.dec2(up2)

        up1 = self.upsample(up2)  # 12x12 -> 24x24
        up1 = torch.cat([up1, x1], dim=1)  # (N, 128+64, 24, 24)
        up1 = self.dec1(up1)

        # Final output
        out = self.last_layer(up1)  # 24x24 -> 24x24 with 6 channels
        return out