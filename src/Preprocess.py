import os
import cv2
import numpy as np
import torch
from torchvision import transforms


class Preprocess:
    def __init__(self, path:str, n_lights:int = 4, width:int = 224, height:int =224, top_light:bool = True):
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
        i = 0
        for image in listdir:
            if image.endswith('.png'):
                number = int(image.split('.')[0])  # Extract the number from the filename
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

    def process_images(self, savefilename:str, n_images:int = 4) -> None:
        """
        Process images for training and testing.
        """
        images = self.load_images(self.path, n_images)
        print('Loaded images shape:', np.array(images).shape)
        images = self.greyscale_images(images)
        print('Greyscale images shape:', np.array(images).shape)
        images = self.layer_images(images, n_images)

        print('Images shape:', images.shape)

        np.save(savefilename + '.npy', images)
    
    def process_flat_images(self, savefilename:str) -> None:
        """
        Process images for training and testing.
        """
        images = self.load_flat_images(self.path)
        images = self.greyscale_images(images)
        print('Loaded images shape:', np.array(images).shape)

        np.save(savefilename + '.npy', images)
        return images
        
        
if __name__ == '__main__':
    preprocess = Preprocess(path='Datasets\Dataset003\Train', n_lights=24, width=224, height=224, top_light=True)
    
    # images = preprocess.process_images('Datasets\Dataset003\Train24Lights', n_images=24)
    images = preprocess.process_flat_images('Datasets\Dataset003\TrainTopLight')
