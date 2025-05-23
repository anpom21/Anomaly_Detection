import torch
import torch.nn as nn
import torch.optim as optim
#from Autoencoder import load_np_data
import tqdm
from pathlib import Path
import numpy as np
import os, shutil
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchsummary import summary
from torch.utils.data.dataset import Subset
import urllib.request
import tarfile
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score
import seaborn as sns

#------------------------------------------------------------------ Load Data ------------------------------------------------------------------
transform_np = transforms.Compose([
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor and divide by 255.0
    transforms.Resize((224,224))  # Resize the image to 224x224 pixels 
])

def load_np_data():
    # load the list containing the four channel images
    data = np.load('carpet/layered_images.npy')

    print("Data array shape: ", data.shape)
    print("First element of data array: ", data[0].shape)

    # Load the list using the Imagefolder dataset class
    for img in data:
        #convert np.array to PIL image
        #img = Image.fromarray(img)
        img = np.moveaxis(img, 0, -1)  # Convert (C, H, W) -> (H, W, C)
        img = transform_np(img)
        #print("img shape: ", img.shape)

    # Split the dataset into training and testing subsets
    # The `torch.utils.data.random_split` function randomly splits a dataset into non-overlapping subsets
    # The first argument `good_dataset` is the dataset to be split
    # The second argument `[0.8, 0.2]` specifies the sizes of the subsets. Here, 80% for training and 20% for testing.
    #train_dataset, test_dataset = torch.utils.data.random_split(good_dataset, [0.75, 0.25])
    train_dataset, test_dataset = torch.utils.data.random_split(data, [0.8, 0.2])
    
    # Print the lengths of the original dataset, training subset, and testing subset
    print("Total number of samples in the original dataset:", len(data))
    print("Number of samples in the training subset:", len(train_dataset))
    print("Number of samples in the testing subset:", len(test_dataset))

    # Assuming train_dataset and test_dataset are PyTorch datasets containing image data and labels
    # Set the batch size
    BS = 5#16

    # Create data loaders for training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=True)

    # Get a batch of images and labels from the training loader
    #image_batch, label_batch = next(iter(train_loader))
    image_batch = next(iter(train_loader))

    # Print the shape of the input images and labels
    print(f'Shape of input images: {image_batch.shape}')
    #print(f'Shape of labels: {label_batch.shape}')

    # Set the figure size
    plt.figure(figsize=(12*4, 48*4))

    # Create a grid of images from the image batch and visualize it
    grid = torchvision.utils.make_grid(image_batch[0:4], padding=5, nrow=4)
    plt.imshow(grid.permute(1, 2, 0))  # Permute dimensions to (height, width, channels) for visualization
    plt.title('Good Samples')  # Set the title of the plot
    plt.show()  # Show the plot

    return train_loader, test_loader

# ----------------------------------------------------------------- ResNet Architecture -----------------------------------------------------------------

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)
        
#----------------------------------------------------------------- ResNet Models -----------------------------------------------------------------  
#       
def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)

def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)

#------------------------------------------------------------------ ResNet Autoencoder ------------------------------------------------------------------

class ResNetAutoencoder(nn.Module):
    def __init__(self, channels=4):
        super(ResNetAutoencoder, self).__init__()

        # Use your custom ResNet50 as encoder (but remove fc & avgpool)
        self.encoder = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1, num_channels=channels)
        
        # Remove classification head
        self.encoder.avgpool = nn.Identity()
        self.encoder.fc = nn.Identity()

        # Decoder: Upsample back to original resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),   # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),    # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),    # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),     # 128 -> 256
            nn.ReLU(),
            nn.Conv2d(64, channels, kernel_size=3, padding=1),        # Output = 4 channels
            nn.Sigmoid()
        )

    def forward(self, x):
        # Only go through encoder layers, not the fc/avgpool
        x = self.encoder.relu(self.encoder.batch_norm1(self.encoder.conv1(x)))
        x = self.encoder.max_pool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        # Decoder
        x = self.decoder(x)
        return x

#------------------------------------------------------------------- ResNet Feature Extractor -------------------------------------------------------------------
# Copilots suggestion
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, channels=4):
        super(ResNetFeatureExtractor, self).__init__()

        # Use your custom ResNet50 as encoder (but remove fc & avgpool)
        self.encoder = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1, num_channels=channels)
        
        # Remove classification head
        self.encoder.avgpool = nn.Identity()
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        # Only go through encoder layers, not the fc/avgpool
        x = self.encoder.relu(self.encoder.batch_norm1(self.encoder.conv1(x)))
        x = self.encoder.max_pool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        return x

# Implementation 
class ResNet2(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3, extract_features=False):
        super(ResNet2, self).__init__()
        self.extract_features = extract_features
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        if self.extract_features:
            self.avgpool = nn.Identity()
            self.fc = nn.Identity()
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if not self.extract_features:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

def ResNet50_2(num_classes, channels=3):
    return ResNet2(Bottleneck, [3,4,6,3], num_classes, channels)

class CustomResNetFeatureExtractor(nn.Module):
    def __init__(self, model):
        super(CustomResNetFeatureExtractor, self).__init__()
        self.model = model
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.features = []

        def hook(module, input, output):
            self.features.append(output)

        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

        self.avg = nn.AvgPool2d(3, stride=1)
        self.resize = nn.AdaptiveAvgPool2d((28, 28))  # match expected shape

    def forward(self, x):
        self.features = []
        with torch.no_grad():
            _ = self.model(x)

        resized_maps = [self.resize(self.avg(fmap)) for fmap in self.features]
        patch = torch.cat(resized_maps, 1)
        return patch

class FeatCAE(nn.Module):
    """Autoencoder."""

    def __init__(self, in_channels=1000, latent_dim=50, is_bn=True):
        super(FeatCAE, self).__init__()

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

# ---------------------------------------------------------------- Train ----------------------------------------------------------------

#Patience = number of epochs before stopping
def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, save_path='best_model.pth', patience=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()  # Reconstruction loss for autoencoders
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # === Training ===
        model.train()
        running_loss = 0.0
        total = 0

        for images in train_loader:
            if isinstance(images, (list, tuple)):
                images = images[0]  # In case it's a tuple like (images, dummy_label)

            images = images.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            #print(f"Input shape: {images.shape}")
            #print(f"Output shape: {outputs.shape}")

            loss = criterion(outputs, images)  # reconstruct input
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total += images.size(0)

        train_loss = running_loss / total
        print(f"Train Loss: {train_loss:.4f}")

        # === Validation ===
        model.eval()
        val_loss = 0.0
        val_total = 0

        with torch.no_grad():
            for images in val_loader:
                if isinstance(images, (list, tuple)):
                    images = images[0]

                images = images.to(device).float()
                outputs = model(images)
                loss = criterion(outputs, images)

                val_loss += loss.item() * images.size(0)
                val_total += images.size(0)

        val_loss /= val_total
        print(f"Val Loss: {val_loss:.4f}")

        # === Check for improvement ===
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ New best model saved at epoch {epoch+1}")
        else:
            epochs_without_improvement += 1
            print(f"⚠️ No improvement for {epochs_without_improvement} epoch(s)")

        # === Early stopping ===
        if epochs_without_improvement >= patience:
            print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs. Best Val Loss: {best_loss:.4f}")
            break

    print(f"\n🎯 Training complete. Best validation loss: {best_loss:.4f}")


# def train_model(model, feature_extractor, train_loader, val_loader, num_epochs=20, lr=0.001, save_path='best_model.pth', patience=5):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     feature_extractor = feature_extractor.to(device)

#     # Enable training on both
#     for param in feature_extractor.parameters():
#         param.requires_grad = True

#     # MSE loss for reconstruction in feature space
#     criterion = nn.MSELoss()

#     # Optimizer for both models
#     optimizer = optim.Adam(
#         list(model.parameters()) + list(feature_extractor.model.parameters()),  # <-- use .model if it's wrapped
#         lr=lr
#     )

#     best_loss = float('inf')
#     epochs_without_improvement = 0

#     for epoch in tqdm(range(num_epochs)):
#         print(f"\nEpoch {epoch + 1}/{num_epochs}")

#         # === Training ===
#         model.train()
#         feature_extractor.train()
#         running_loss = 0.0
#         total = 0

#         for images in train_loader:
#             if isinstance(images, (list, tuple)):
#                 images = images[0]
#             images = images.to(device).float()

#             # Forward through feature extractor (no torch.no_grad!)
#             features = feature_extractor(images)
#             outputs = model(features)

#             loss = criterion(outputs, features)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * images.size(0)
#             total += images.size(0)

#         train_loss = running_loss / total
#         print(f"Train Loss: {train_loss:.4f}")

#         # === Validation ===
#         model.eval()
#         feature_extractor.eval()
#         val_loss = 0.0
#         val_total = 0

#         with torch.no_grad():
#             for images in val_loader:
#                 if isinstance(images, (list, tuple)):
#                     images = images[0]
#                 images = images.to(device).float()

#                 features = feature_extractor(images)
#                 outputs = model(features)

#                 loss = criterion(outputs, features)

#                 val_loss += loss.item() * images.size(0)
#                 val_total += images.size(0)

#         val_loss /= val_total
#         print(f"Val Loss: {val_loss:.4f}")

#         # === Save best model ===
#         if val_loss < best_loss:
#             best_loss = val_loss
#             epochs_without_improvement = 0
#             torch.save({
#                 'featcae_state_dict': model.state_dict(),
#                 'resnet_state_dict': feature_extractor.model.state_dict()
#             }, save_path)
#             print(f"✅ New best model saved at epoch {epoch+1}")
#     #    else:
#     #        epochs_without_improvement += 1
#     #        print(f"⚠️ No improvement for {epochs_without_improvement} epoch(s)")

#     #    if epochs_without_improvement >= patience:
#     #        print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs. Best Val Loss: {best_loss:.4f}")
#     #        break

#     print(f"\n🎯 Training complete. Best validation loss: {best_loss:.4f}")


#------------------------------------------------------------------ Test ------------------------------------------------------------------

# Create a custom dataset with images and empty labels
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0  # Ensure float32
        return torch.tensor(img), self.labels[idx]
        #return self.images[idx], self.labels[idx]  # Return (image, empty list)


# load test dataset
def load_test():
    data = np.load('carpet/test_layered_images.npy')

    print("Data array shape: ", data.shape)
    print("First element of data array: ", data[0].shape)

    # Load the list using the Imagefolder dataset class
    for img in data:
        #convert np.array to PIL image
        img = np.moveaxis(img, 0, -1)  # Convert (C, H, W) -> (H, W, C)
        img = transform_np(img)
        print("img shape: ", img.shape, " ", img.dtype)
    
    # Create dummy labels (empty lists)
    labels = [[0] for _ in range(len(data))]  # Each image gets an empty list as a label

    dataset = CustomDataset(data, labels)

    return dataset

# process the test dataset
def test(dataset, model):
    #load images from the dataset
    data = torch.stack([img for img, _ in dataset])

    with torch.no_grad():
        data = data.cuda()
        recon = model(data)
        
    recon_error =  ((data-recon)**2).mean(axis=1)
    print(recon_error.shape)
        
    for i in range(98):
        #save the error image
        error_array = recon_error[i][0:-10,0:-10].cpu().numpy()
        # Normalize to [0,255] range for proper image saving
        error_array = (error_array - error_array.min()) / (error_array.max() - error_array.min())  # Normalize to [0,1]
        error_array = (error_array * 255).astype(np.uint8)  # Scale to [0,255] and convert to uint8
        error_img = Image.fromarray(error_array, mode="L")  # Convert to grayscale
        error_img.save(f"test/error_{i}.png")
        #recon_error_img = Image.fromarray(recon_error[i][0:-10,0:-10].cpu().numpy())
        #recon_error_img.save(f"recon_error_{i}.png")
        #Display the error image
        plt.imshow(error_array, cmap='jet')
        plt.title(f"Error Image {i}")
        plt.axis('off')
        plt.show()
       
# def test(dataset, model, feature_extractor):
#     #load images from the dataset
#     data = torch.stack([img for img, _ in dataset])

#     with torch.no_grad():
#         data = data.cuda()
#         features = feature_extractor(data)
#         recon = model(features)
        
#     recon_error =  ((features-recon)**2).mean(axis=1)
#     print(recon_error.shape)
        
#     for i in range(98):
#         #save the error image
#         error_array = recon_error[i][0:-10,0:-10].cpu().numpy()
#         # Normalize to [0,255] range for proper image saving
#         error_array = (error_array - error_array.min()) / (error_array.max() - error_array.min())  # Normalize to [0,1]
#         error_array = (error_array * 255).astype(np.uint8)  # Scale to [0,255] and convert to uint8
#         error_img = Image.fromarray(error_array, mode="L")  # Convert to grayscale
#         #error_img.save(f"test/error_{i}.png")
#         #Display the error image
#         plt.imshow(error_array, cmap='gray')
#         plt.title(f"Error Image {i}")
#         plt.axis('off')
#         plt.show()

# ----------------------------------------------------------------- Main -----------------------------------------------------------------
#Load dataset
train_loader, test_loader = load_np_data()

#load model
#model = ResNet50(num_classes=2, channels=4)

# Instantiate your custom ResNet (as a feature extractor)
#custom_resnet = ResNet50(num_classes=10, channels=4)  # or 3 if RGB
#custom_resnet.extract_features = True  # disable classifier
#feature_extractor = CustomResNetFeatureExtractor(custom_resnet).cuda()
# Feature autoencoder
#model = FeatCAE(in_channels=1536, latent_dim=100).cuda()

#Instantiate the ResNet autoencoder
model = ResNetAutoencoder(channels=4).cuda()

# Optionally: fine-tune ResNet too
# optimizer = torch.optim.Adam(list(model.parameters()) + list(custom_resnet.parameters()), lr=0.001)


#Train model
#train_model(model, train_loader, test_loader, num_epochs=50, lr=0.001, save_path='best_model.pth', patience=5)
train_model(model, train_loader=train_loader, val_loader=test_loader, num_epochs=100, lr=0.001, save_path='ResNet50Autoencoder.pth', patience=100)
#train_model(model, feature_extractor, train_loader, test_loader, num_epochs=1000, lr=0.001, save_path='OldDataResNet50AutoEnc.pth', patience=500)

# Load trained model
#checkpoint = torch.load('OldDataResNet50AutoEnc.pth')
#model.load_state_dict(checkpoint['featcae_state_dict'])
#feature_extractor.model.load_state_dict(checkpoint['resnet_state_dict'])
checkpoint = torch.load('ResNet50Autoencoder.pth')
model.load_state_dict(checkpoint)

test_dataset = load_test()
#test(test_dataset, model, feature_extractor)
test(test_dataset, model)