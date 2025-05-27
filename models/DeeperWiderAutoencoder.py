import torch
import torch.nn as nn

class DeeperWiderAutoencoder(nn.Module):
    def __init__(self, input_channels=4):
        self.channels = input_channels
        super(DeeperWiderAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 256 -> 128
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 128 -> 64
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 64 -> 32
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 32 -> 16
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, self.channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        #print(f"Encoder output shape: {x.shape}")
        x = self.decoder(x)
        return x
    
if __name__ == "__main__":
    model = DeeperWiderAutoencoder(input_channels=4)
    print(model)
    x = torch.randn(1, 4, 224, 224)  # Example input
    output = model(x)
    print(output.shape)  # Should be (1, 4, 256, 256)