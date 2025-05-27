import torch
import torch.nn as nn

class WiderAutoencoder(nn.Module):
    def __init__(self, input_channels=4):
        self.channels = input_channels
        super(WiderAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 256, kernel_size=4),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=4),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, output_padding=1 ),
            nn.ReLU(),
            nn.ConvTranspose2d(256, self.channels, kernel_size=5, stride=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        #print(f"Encoder output shape: {x.shape}")
        x = self.decoder(x)
        return x
    
if __name__ == "__main__":
    model = WiderAutoencoder(input_channels=4)
    print(model)
    x = torch.randn(1, 4, 224, 224)  # Example input
    output = model(x)
    print(output.shape)  # Should be (1, 4, 256, 256)