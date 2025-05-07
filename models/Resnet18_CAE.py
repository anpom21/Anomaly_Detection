import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from src.Dataloader import Dataloader

class resnet18_feature_extractor(torch.nn.Module):
    def __init__(self, input_channels = 4):
        """This class extracts the feature maps from a pretrained Resnet model."""
        super(resnet18_feature_extractor, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        weights_avg = torch.mean(self.model.conv1.weight.clone(), dim=1, keepdim=True)

        weights = [weights_avg for _ in range(input_channels)]  # Create a list of weights for each input channel
        weights = torch.cat(weights, dim=1)  # Concatenate along the channel dimension

        self.model.conv1.weight = torch.nn.Parameter(weights)  # Assign the new weights to the conv1 layer
        self.model.conv1.in_channels = input_channels  # Update the in_channels attribute

        # input_weights_avg = torch.mean(ResNet18_Weights.DEFAULT.meta['input_size'][0])
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        

        # Hook to extract feature maps
        def hook(module, input, output) -> None:
            """This hook saves the extracted feature map on self.featured."""
            self.features.append(output)

        self.model.layer2[-1].register_forward_hook(hook)            
        self.model.layer3[-1].register_forward_hook(hook) 

    def forward(self, input):

        self.features = []
        with torch.no_grad():
            _ = self.model(input)

        self.avg = torch.nn.AvgPool2d(3, stride=1)
        fmap_size = self.features[0].shape[-2]         # Feature map sizes h, w
        self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)

        resized_maps = [self.resize(self.avg(fmap)) for fmap in self.features]
        patch = torch.cat(resized_maps, 1)            # Merge the resized feature maps

        return patch
    

class resnet18_autoencoder(nn.Module):
    """Autoencoder."""

    def __init__(self, in_channels=1000, latent_dim=50, is_bn=True):
        super(resnet18_autoencoder, self).__init__()

        layers = []
        layers += [nn.Conv2d(in_channels, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)]

        self.encoder = nn.Sequential(*layers)

        # if 1x1 conv to reconstruct the rgb values, we try to learn a linear combination
        # of the features for rgb
        layers = []
        layers += [nn.Conv2d(latent_dim, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, in_channels, kernel_size=1, stride=1, padding=0)]
        # layers += [nn.ReLU()]

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

if __name__ == "__main__":
    from torchsummary import summary
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_images = 4
    model = resnet18_feature_extractor(n_images).to(device)
    autoencoder = resnet18_autoencoder().to(device)

    dataloader = Dataloader("Datasets/Dataset002")
    train_loader, vali_loader, test_loader = dataloader.load_train_vali_test_dataloaders_with_n_images(n_images = 4, BS=16)

    import matplotlib.pyplot as plt

    # Get the first batch of images
    images = next(iter(train_loader))
    # Select the first image
    first_image = images[0].unsqueeze(0)  # Remove the batch dimension

    model.eval()
    with torch.no_grad():
        feature_maps = model(first_image.to(device))
        print("Feature maps shape:", feature_maps.shape)

