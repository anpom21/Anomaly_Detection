import torch
import numpy as np
import matplotlib.pyplot as plt
import tarfile
import os
import torchvision
import torchvision.transforms.functional
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import urllib.request

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


# Define the transformation pipeline using torchvision.transforms.Compose
transform = transforms.Compose([
    transforms.Resize((224,224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor()          # Convert the image to a PyTorch tensor and divide by 255.0
])

transform_np = transforms.Compose([
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor and divide by 255.0
    transforms.Resize((224,224)),   # Resize the image to 224x224 pixels
    transforms.GaussianBlur(kernel_size=5, sigma=2.0) # Select the first 4 channels
])

class Datasets:
    # load the data and split it into training and testing sets
    def load_data():
        # Download and extract the MVTec AD dataset
        if not os.path.exists('carpet'):
            urllib.request.urlretrieve("https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz", "carpet.tar.xz")

            with tarfile.open('carpet.tar.xz') as f:
                f.extractall('.')

        # Links to other objects from MVTEC AD dataset
        # "bottle": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz",
        # "cable": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz",
        # "capsule": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz",
        # "carpet": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz",
        # "grid": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz",
        # "hazelnut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz",
        # "leather": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz",
        # "metal_nut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz",
        # "pill": "https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz",
        # "screw": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz",
        # "tile": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz",
        # "toothbrush": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz",
        # "transistor": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz",
        # "wood": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz",
        # "zipper": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz"

        # Define the path to the directory containing the train images
        train_image_path = 'carpet/train'

        # Load the train dataset using the ImageFolder dataset class
        # ImageFolder is a PyTorch dataset class for loading images from a directory
        # It automatically loads images from subdirectories and applies transformations to them
        # In this case, 'transform' is a transformation applied to each image in the dataset
        # It preprocesses the images before they are used for training
        good_dataset = ImageFolder(root=train_image_path, transform=transform)

        # Access a sample from the dataset
        # In this case, we're accessing the first sample
        # x contains the preprocessed image data
        # y contains the corresponding label (class index)
        x, y = good_dataset[0]

        # Print the shape of the preprocessed image data (x) and its corresponding label (y)
        print("Image Shape:", x.shape)
        print("Label:", y)

        # Split the dataset into training and testing subsets
        # The `torch.utils.data.random_split` function randomly splits a dataset into non-overlapping subsets
        # The first argument `good_dataset` is the dataset to be split
        # The second argument `[0.8, 0.2]` specifies the sizes of the subsets. Here, 80% for training and 20% for testing.
        train_dataset, test_dataset = torch.utils.data.random_split(good_dataset, [0.8, 0.2])

        # Print the lengths of the original dataset, training subset, and testing subset
        print("Total number of samples in the original dataset:", len(good_dataset))
        print("Number of samples in the training subset:", len(train_dataset))
        print("Number of samples in the testing subset:", len(test_dataset))

        # Assuming train_dataset and test_dataset are PyTorch datasets containing image data and labels
        # Set the batch size
        BS = 16

        # Create data loaders for training and testing datasets
        train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=True)

        # Get a batch of images and labels from the training loader
        image_batch, label_batch = next(iter(train_loader))

        # Print the shape of the input images and labels
        print(f'Shape of input images: {image_batch.shape}')
        print(f'Shape of labels: {label_batch.shape}')

        # Set the figure size
        plt.figure(figsize=(12*4, 48*4))

        # Create a grid of images from the image batch and visualize it
        grid = torchvision.utils.make_grid(image_batch[0:4], padding=5, nrow=4)
        plt.imshow(grid.permute(1, 2, 0))  # Permute dimensions to (height, width, channels) for visualization
        plt.title('Good Samples')  # Set the title of the plot
        plt.show()  # Show the plot

        return train_loader, test_loader

    def load_np_data(train = True):
        # load the list containing the four channel images
        if train:
            data = np.load('carpet/layered_images2.npy')
        else:
            data = np.load('carpet/test_layered_images2.npy')

        print("Data array shape: ", data.shape)
        print("First element of data array: ", data[0].shape)

        # Load the list using the Imagefolder dataset class
        for img in data:
            #convert np.array to PIL image
            #img = Image.fromarray(img)
            img = np.moveaxis(img, 0, -1)  # Convert (C, H, W) -> (H, W, C)
            img = transform_np(img)
            print("img shape: ", img.shape, " ", img.dtype)
        
        # Create dummy labels (empty lists)
        labels = [[0] for _ in range(len(data))]  # Each image gets an empty list as a label

        dataset = CustomDataset(data, labels)

        if train != True:
            # load the first three images from the dataset into a tensor
            test = torch.stack([dataset[0][0], dataset[1][0], dataset[2][0]])
            return test

        # Split the dataset into training and testing subsets
        # The `torch.utils.data.random_split` function randomly splits a dataset into non-overlapping subsets
        # The first argument `good_dataset` is the dataset to be split
        # The second argument `[0.8, 0.2]` specifies the sizes of the subsets. Here, 80% for training and 20% for testing.
        #train_dataset, test_dataset = torch.utils.data.random_split(good_dataset, [0.75, 0.25])
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
        
        # Print the lengths of the original dataset, training subset, and testing subset
        print("Total number of samples in the original dataset:", len(dataset))
        print("Number of samples in the training subset:", len(train_dataset))
        print("Number of samples in the testing subset:", len(test_dataset))

        # Assuming train_dataset and test_dataset are PyTorch datasets containing image data and labels
        # Set the batch size
        BS = 5#16

        # Create data loaders for training and testing datasets
        train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=True)

        # Get a batch of images and labels from the training loader
        image_batch, label_batch = next(iter(train_loader))
        #image_batch = next(iter(train_loader))

        # Print the shape of the input images and labels
        print(f'Shape of input images: {image_batch.shape}')
        #print(f'Shape of labels: {label_batch.shape}')
        #print(f'Labels: {label_batch}')

        # Set the figure size
        plt.figure(figsize=(12*4, 48*4))

        # Create a grid of images from the image batch and visualize it
        grid = torchvision.utils.make_grid(image_batch[0:4], padding=5, nrow=4)
        plt.imshow(grid.permute(1, 2, 0))  # Permute dimensions to (height, width, channels) for visualization
        plt.title('Good Samples')  # Set the title of the plot
        plt.show()  # Show the plot

        return train_loader, test_loader
    
    # load test dataset
    def load_test():
        data = np.load('carpet/test_layered_images2.npy')

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