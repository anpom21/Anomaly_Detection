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
from torch.utils.data import DataLoader
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


# ---------------------------------------------------------------- Train ----------------------------------------------------------------

#Patience = number of epochs before stopping
def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, save_path='best_model.pth', patience=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()  # Reconstruction loss for autoencoders
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
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
            print(f"Input shape: {images.shape}")
            print(f"Output shape: {outputs.shape}")

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



# ----------------------------------------------------------------- Main -----------------------------------------------------------------
#Load dataset
train_loader, test_loader = load_np_data()

#load model
model = ResNet50(num_classes=2, channels=4)

#Train model
train_model(model, train_loader, test_loader, num_epochs=50, lr=0.001, save_path='best_model.pth', patience=5)
