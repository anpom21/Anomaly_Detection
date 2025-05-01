import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from models.DeeperWiderAutoencoder import DeeperWiderAutoencoder

from src.Dataloader import Dataloader
from src.Trainer import Trainer

def load_datasets_with_n_lightsources(n_lights:int, path:str):
    '''
    Load the datasets with a specific number of lightsources.
    '''
    dataset = Dataloader(path=path, n_lights=n_lights, width=224, height=224, top_light=True)
    images = dataset.get_images(n_images=n_lights)
    return images

def main():
    #model.load_state_dict(torch.load("models\Trained_models\DeeperWider_autoencoder.pth"))
    #Loop that trains the model with different amount of lightsources:
    print("Loading dataset")
    dataset = Dataloader(path='Datasets/Dataset003/Train', n_lights=24, width=224, height=224, top_light=True)
    print("Dataset loaded")
    i = 0 
    print("Starting test")
    while i <= 24:
        i += 4
        print(f"Testing with {i} lightsources")
        #Load the data
        #TODO: This is a list of images, we need them to be loaded into a train and validation dataloader
        train_loader, test_loader = dataset.load_train_test_dataloaders_with_n_images(train_path='Datasets/Dataset003/Train', test_path='Datasets/Dataset003/Test', n_images=i)

        #Load the model
        model = DeeperWiderAutoencoder(input_channels=i)

        Trainer.train_model(model=model, train_loader=train_loader,
                            val_loader=test_loader, num_epochs=200, 
                            lr=0.001, save_path=f"models/Trained_models/DeeperWider_autoencoder_{i}.pth", 
                            patience=50)


main()


        
        
