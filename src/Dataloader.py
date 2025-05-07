import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import random_split
import yaml



class Dataloader:
    def __init__(self, path:str):
        '''
        Preprocess images for training and testing.
        '''
        self.path = path
        with open(path + "/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.n_lights = config['n_lights']
        self.width = config['image_shape'][0]
        self.height = config['image_shape'][1]
        self.flat_light = config['flat_light']
        self.n_images = self.n_lights + self.flat_light
        self.n_samples = config['n_samples']

        self.n_abnormals = config['n_abnormals']
        self.n_normals = config['n_normals']

    def select_image_indexes(self, n:int):
        indexes = []
        for i in range(n):
            indexes.append(i*self.n_lights//n)
        return indexes
    
    def greyscale_images(self, images):
        greyscale_images = []
        for image in images:
            greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            greyscale_images.append(greyscale_image)
        return greyscale_images
    
    def load_images(self, path, n):
        '''
        Load images from the dataset
        '''
        image_paths = []
        listdir = os.listdir(path)

        indexes = self.select_image_indexes(n)

        # Check if the directory is empty
        assert len(listdir) > 0, 'No images found in the directory'
        i = 0
        for image in listdir:
            if image.endswith('.png'):
                number = int(image.split('.')[0][-4:])  # Extract the number from the filename
                if number % self.n_images in indexes:
                    image_paths.append(os.path.join(path, image))
            
            # i+= 1
            # if i >= 100:
            #     break

        images = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            images.append(image)

        return images

    def load_flat_images(self, path):
        '''
        Load flat images from the dataset
        '''
        image_paths = []
        listdir = os.listdir(path)
        # Check if the directory is empty
        assert len(listdir) > 0, 'No images found in the directory'
        i = 0
        for image in listdir:
            if image.endswith('.png'):
                number = int(image.split('.')[0])  # Extract the number from the filename
                if number % self.n_images == self.n_images - 1:
                    image_paths.append(os.path.join(path, image))
            
            # i+= 1
            # if i >= 100:
            #     break

        images = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            images.append(image)
        return images
    
    def layer_images(self, images, n) -> np.ndarray:
        # Reshape the images list into a 2D list with n columns
        reshaped_images = [images[i:i + n] for i in range(0, len(images), n)]
        reshaped_images = np.array(reshaped_images)

        return reshaped_images

    def get_images(self, path:str, n_images:int = 4) -> np.ndarray:
        """
        Process images for training and testing.
        """
        images = self.load_images(path, n_images)
        # print('Loaded images shape:', np.array(images).shape)
        images = self.greyscale_images(images)
        # print('Greyscale images shape:', np.array(images).shape)
        images = self.layer_images(images, n_images)

        # print('Images shape:', images.shape)

        
        return images
    
    def load_flat_data(self, savefilename:str) -> None:
        """
        Process images for training and testing.
        """
        images = self.load_flat_images(self.path)
        images = self.greyscale_images(images)
        # print('Loaded images shape:', np.array(images).shape)

        
        return images
    
    def load_train_vali_test_dataloaders_with_n_images(self, n_images: int = 4, trainSplit: float = 0.8, BS: int = 16) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Process images for training and testing.
        """
        Dataset = self.get_images(self.path + "/Train", n_images)
        Dataset = Dataset / 255.0
        Dataset = torch.tensor(Dataset, dtype=torch.float32)

        # Split the dataset into train and test sets
        train_size = int(trainSplit * len(Dataset))
        test_size = len(Dataset) - train_size
        train_dataset, vali_dataset = random_split(Dataset, [train_size, test_size])

        # Create DataLoaders for training and testing datasets
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BS, shuffle=True)
        vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=BS, shuffle=True)
        test_loader = self.load_test_dataloader(n_images=n_images, BS=BS)

        return train_loader, vali_loader, test_loader
    
    def load_test_dataloader(self, n_images: int = 4, BS: int = 16) -> torch.utils.data.DataLoader:
        """
        Load test images and labels them as defect or normal.
        """
        test_path = self.path + "/Test"
        images = self.load_images(self.path + "/Test", n_images)
        images = self.greyscale_images(images)
        images = self.layer_images(images, n_images)
        # labels = np.concatenate((np.ones(self.n_abnormals), np.zeros(self.n_normals))) # Alternative way to create labels
        labels = self.load_labels(test_path)

        assert len(images) == len(labels), "Mismatch in number of images and labels"

        # Normalize and convert to tensors
        images = np.array(images) / 255.0
        images = torch.tensor(images, dtype=torch.float32)  # Add channel dimension
        labels = torch.tensor(labels, dtype=torch.long) 

        # Create a dataset and DataLoader
        dataset = torch.utils.data.TensorDataset(images, labels)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=BS, shuffle=False)

        return test_loader
    
    def load_labels(self, path:str) -> np.ndarray:
        """
        Load labels from the filename of the images in the directory.
        """
        labels = []
        listdir = os.listdir(path)
        # Check if the directory is empty
        assert len(listdir) > 0, 'No images found in the directory'
        for i in range(len(listdir)):
            if listdir[i].endswith('.png') and i % self.n_images == 0:
                # Append the filenae of the image without numbers to labels
                filename = listdir[i].split('.')[0]  # Get the filename without the extension
                if filename.isdigit():  # If the filename is just numbers
                    labels.append(0)  # No anomaly
                elif filename.startswith('good'):  # If the filename starts with 'good'
                    labels.append(0)  # No anomaly
                elif filename.startswith('defect'):  # If the filename starts with 'defect'
                    labels.append(1)  # Anomaly

        return labels
        
        
if __name__ == '__main__':
    preprocess = Dataloader(path='Datasets/Dataset004')
    train_loader, vali_loader, test_loader = preprocess.load_train_vali_test_dataloaders_with_n_images(n_images=4, trainSplit=0.8, BS=16)

    # print('Train loader:', train_loader.dataset.dataset.shape)
    # print('Test loader:', test_loader.dataset.tensors[0].shape)
    # print('labels:', test_loader.dataset.tensors[1].shape)
    print(preprocess.n_images)
    images = preprocess.get_images(preprocess.path + "\Train", n_images=4)
    print("Images shape:", np.array(images).shape)
    images = torch.tensor(images, dtype=torch.float32)
    image = images[0].unsqueeze(0)
    print("Input shape:", image.shape)


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(image[0].cpu().detach().numpy()[0])
    ax[1].imshow(image[0].cpu().detach().numpy()[1])
    ax[2].imshow(image[0].cpu().detach().numpy()[2])
    ax[3].imshow(image[0].cpu().detach().numpy()[3])
    plt.show()