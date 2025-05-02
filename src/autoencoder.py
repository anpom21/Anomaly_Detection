




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
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
            
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, n_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        self.encoded = encoded
        decoded = self.decoder(encoded)
        return decoded
    
    def forward_print(self, x):
        print(f"Input shape: {x.shape}")

        # Encoder
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            print(f"After encoder layer {i} ({layer.__class__.__name__}): {x.shape}")
        encoded = x

        # Decoder
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            print(f"After decoder layer {i} ({layer.__class__.__name__}): {x.shape}")
        decoded = x

        return decoded
    





if __name__ == "__main__":
    from src.Dataloader import Dataloader
    process = Dataloader(path='Datasets\Dataset002\Train', n_lights=4, width=224, height=224, top_light=False)
    images = process.get_images(n_images=4)

    images = torch.tensor(images, dtype=torch.float32)
    image = images[0]
    model = Autoencoder(n_channels=4)
    output = model(image)
    output = model.forward_print(image)

    print("Input shape:", image.shape)
    print("Output shape:", output.shape)
    print("Encoded shape:", model.encoded.shape)


    # print(image[0].detach().numpy())
    from matplotlib import pyplot as plt
    plt.subplot(1, 3, 1)
    plt.title('Input Image')
    plt.imshow(image[0].detach().numpy(), cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('Encoded Image')
    plt.imshow(model.encoded[0].detach().numpy(), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Output Image')
    plt.imshow(output[0].detach().numpy(), cmap='gray')
    plt.show()
