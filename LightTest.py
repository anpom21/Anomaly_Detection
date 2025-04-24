import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from models import DeeperWiderAutoencoder

from src.Dataloader import Dataloader



def main():
    model = DeeperWiderAutoencoder()
    #model.load_state_dict(torch.load("models\Trained_models\DeeperWider_autoencoder.pth"))
    #Loop that trains the model with different amount of lightsources:
    i = 0
    while i <= 24:
        i += 4
        print(f"Testing with {i} lightsources")
        #Load the data
        #TODO: This is a list of images, we need them to be loaded into a train and validation dataloader
        data = Dataloader(path='Datasets\Dataset003\Train', n_lights=i, width=224, height=224, top_light=False)

