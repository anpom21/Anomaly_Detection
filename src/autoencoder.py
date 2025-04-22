




import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, n_channels):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, n_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    





if __name__ == "__main__":
    from Dataloader import Dataloader
    process = Dataloader(path='Datasets\Dataset003\Train', n_lights=24, width=224, height=224, top_light=True)
    images = process.process_images('Datasets\Dataset003\Train24Lights', n_images=24)
    print('Images shape:', images.shape)

    images = torch.tensor(images, dtype=torch.float32)
    # images = images.permute(0, 3, 1, 2)  # Change shape to (batch_size, channels, height, width)
