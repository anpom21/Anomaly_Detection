import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import random_split


class Dataloader:
    def __init__(self, path:str, n_lights:int = 24, width:int = 224, height:int =224, top_light:bool = True):
        '''
        Preprocess images for training and testing.
        '''
        self.path = path
        self.n_lights = n_lights
        self.width = width
        self.height = height
        self.top_light = top_light
        self.n_images = n_lights + top_light

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

        for image in listdir:
            if image.endswith('.png'):
                filename = image.split('.')[0]  # Extract the filename without the extension
                if filename.isdigit():  # If the filename is numeric
                    number = int(filename)
                    if number % self.n_images in indexes:
                        image_paths.append(os.path.join(path, image))
                elif filename.startswith('good'):  # Handle filenames starting with 'good'
                    number = int(filename[len('good'):])  # Extract the numeric part after 'good'
                    if number % self.n_images in indexes:
                        image_paths.append(os.path.join(path, image))
                elif filename.startswith('defect'):  # Handle filenames starting with 'defect'
                    number = int(filename[len('defect'):])  # Extract the numeric part after 'defect'
                    if number % self.n_images in indexes:
                        image_paths.append(os.path.join(path, image))

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
    
    def load_train_test_dataloaders_with_n_images(self, train_path:str, test_path:str, n_images:int = 4, trainSplit = 0.8, BS=16) -> tuple:
        """
        Process images for training and testing.
        """
        # load train dataset
        print("loading train dataset")
        train_Dataset = self.get_images(path=train_path, n_images=n_images)
        train_Dataset = train_Dataset/ 255.0
        train_Dataset = torch.tensor(train_Dataset, dtype=torch.float32)

        # load test dataset and labels
        print("loading test dataset")
        test_Dataset = self.get_images(path=test_path, n_images=n_images)
        test_Dataset = test_Dataset/ 255.0
        test_Dataset = torch.tensor(test_Dataset, dtype=torch.float32)
        test_labels = self.load_labels(path=test_path)
        test_labels = torch.tensor(test_labels, dtype=torch.float32)

        assert len(test_Dataset) == len(test_labels), "Mismatch between test images and labels!"
        test_dataset = torch.utils.data.TensorDataset(test_Dataset, test_labels)

        print(f"Train Dataset Shape: {train_Dataset.shape}")
        print(f"Test Dataset Shape: {test_Dataset.shape}")
        print(f"Test Labels Shape: {test_labels.shape}")
        # Split the dataset into train and test sets
        #train_size = int(trainSplit * len(Dataset))
        #test_size = len(Dataset) - train_size
        #train_dataset, test_dataset = random_split(Dataset, [train_size, test_size])

        # Create DataLoaders for training and testing datasets
        train_loader = torch.utils.data.DataLoader(train_Dataset, batch_size=BS, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BS, shuffle=True)
        vali_loader = 0
        return train_loader, vali_loader, test_loader
        
        
if __name__ == '__main__':
    preprocess = Dataloader(path='Datasets/Dataset003/Train', n_lights=24, width=224, height=224, top_light=True)
    preprocess.get_images(4)
    # images = preprocess.process_images('Datasets\Dataset003\Train24Lights', n_images=24)
    # np.save('Datasets\Dataset003\Train24Lights' + '.npy', images)
    # images = preprocess.process_flat_images('Datasets\Dataset003\TrainTopLight')
    # np.save('Datasets\Dataset003\TrainTopLight' + '.npy', images)

    # for i in range(1,25):
    #     print(preprocess.select_image_indexes(i))