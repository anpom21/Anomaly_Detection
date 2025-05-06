import torch
import torch.nn as nn

"""
This file contains two autoencoders trying to capture more high frequency details. One is just a much wider autoencoder with less pooling.
The other one is a U-Net style autoencoder with skip connections, less pooling, wider layers, Dilated convolutions, Strided conv down-sampling and Laplacian- or perceptual losses.
Truely a ChatGPT masterpiece.
"""

class TruelyWiderAutoencoder(nn.Module):
    def __init__(self, input_channels=4):
        self.channels = input_channels
        super(TruelyWiderAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 512, kernel_size=3, padding=1),  # Wider with more channels
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),  # Wider and deeper
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # Downsample
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),  # Even wider
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # Downsample
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 2048, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, self.channels, kernel_size=3, padding=1),  # Final layer to match input channels
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class HighFreqUNetAE(nn.Module):
    def __init__(self, in_channels=4, base_ch=256):
        super().__init__()
        # Encoder: wider + strided conv
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch,    kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch,      base_ch,    kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Conv2d(base_ch, base_ch*2, kernel_size=4, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # dilation=2 => padding=2 (not padding=1!)
            nn.Conv2d(base_ch*2, base_ch*2, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, kernel_size=4, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch*8, base_ch*8, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
        )

        # Decoder + skips
        self.up2 = nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch*4*2, base_ch*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch*4,   base_ch*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch*2*2, base_ch*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch*2,   base_ch,   kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Final projection
        self.final = nn.Conv2d(base_ch, in_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        # Bottleneck
        b  = self.bottleneck(d2)

        # Decoder + skip connections
        u2 = self.up2(b)
        c2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(c2)

        u1 = self.up1(d2)
        c1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(c1)

        out = self.final(d1)
        return torch.sigmoid(out)
