import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import os

from models.Autoencoder import Autoencoder
from models.DeeperAutoencoder import DeeperAutoencoder
from models.DeeperWiderAutoencoder import DeeperWiderAutoencoder
from models.NarrowerAutoencoder import NarrowerAutoencoder
from models.TruelyWiderAutoecoder import TruelyWiderAutoencoder, HighFreqUNetAE
from models.WiderAutoencoder import WiderAutoencoder
from models.ResNetAutoencoder import ResNetAutoencoder

from src.Dataloader import Dataloader
from src.Trainer import Trainer

import matplotlib as mpl
import pandas as pd

# Enable LaTeX rendering TODO: Install latex either MiKTeX or TeX Live
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'  # LaTeX default font is Computer Modern (serif)

# Create a list of models to test
models = [
    "Autoencoder",
    "DeeperAutoencoder",
    "DeeperWiderAutoencoder",
    "NarrowerAutoencoder",
    "WiderAutoencoder",
    "ResNetAutoencoder",
]

if __name__ == "__main__":
    dataset_path = 'Datasets/IRL_4_channel_dataset'
    display = False

    for model_name in models:
        # Set amount of lightsources
        i = 4
        #i = 4

        Fig_SavePath = f"RW_Results/IRL_4_channel_Images/{model_name}/"

        # Load the model
        if model_name == "Autoencoder":
            model = Autoencoder(input_channels=i)
        elif model_name == "DeeperAutoencoder":
            model = DeeperAutoencoder(input_channels=i)
        elif model_name == "DeeperWiderAutoencoder":
            model = DeeperWiderAutoencoder(input_channels=i)
        elif model_name == "NarrowerAutoencoder":
            model = NarrowerAutoencoder(input_channels=i)
        elif model_name == "TruelyWiderAutoencoder":
            model = TruelyWiderAutoencoder(input_channels=i, base_ch=128)
        elif model_name == "HighFreqUNetAE":
            model = HighFreqUNetAE(in_channels=i, base_ch=128)
        elif model_name == "WiderAutoencoder":
            model = WiderAutoencoder(input_channels=i)
        elif model_name == "ResNetAutoencoder":
            model = ResNetAutoencoder(channels=i)

        modelName =f"{model_name}_{i}"
        
        # load the dataset
        dataset = Dataloader(dataset_path)
        train_loader, val_loader, test_loader = dataset.load_train_vali_test_dataloaders_with_n_images(n_images=i, trainSplit=0.8, BS=1)

        # Load the model
        model.load_state_dict(torch.load(f"models/Trained_models/RW_Test/IRL_4_channel/{model_name}_{i}.pth"))
        model.eval()

        # Calculate Threshold
        Threshold = Trainer.get_maxPixelThreshold(model=model, train_loader=train_loader)
        print(f"Threshold for model {model_name} with {i} lightsources: {Threshold}")

        # compare the images
        Trainer.compare_images(model=model, test_loader=test_loader, threshold=Threshold, channels=i, thresholdType="maxPixel", FigSavePath=Fig_SavePath, ModelName=modelName, display=display)