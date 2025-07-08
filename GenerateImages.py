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
from models.TruelyWiderAutoecoder import SparserDownSampleAutoencoder, HighFreqUNetAE
from models.WiderAutoencoder import WiderAutoencoder
from models.ResNetAutoencoder import ResNetAutoencoder

from src.Dataloader import Dataloader
from src.Trainer import Trainer

import matplotlib as mpl
import pandas as pd

# Enable LaTeX rendering TODO: Install latex either MiKTeX or TeX Live
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'  # LaTeX default font is Computer Modern (serif)

# TODO: Train the last two models and test everyting agian

# Create a list of models to test
models = [
    "Autoencoder",
    "DeeperAutoencoder",
    "DeeperWiderAutoencoder",
    "NarrowerAutoencoder",
    "WiderAutoencoder",
    "ResNetAutoencoder",
    "SparserDownSampleAutoencoder",
    "HighFreqUNetAE",
]

if __name__ == "__main__":
    #dataset_path = 'Datasets/IRL_3_channel_dataset'
    dataset_path = 'Datasets/IRL_4_channel_dataset'
    #dataset_path = 'Datasets/Dataset004'
    Fig_SavePath = "ReconstructionExamples/RW4/"
    display = False
    lights = 4
    rw = True

    for model_name in models:
        # Load the model
        if model_name == "Autoencoder":
            model = Autoencoder(input_channels=lights)
        elif model_name == "DeeperAutoencoder":
            model = DeeperAutoencoder(input_channels=lights)
        elif model_name == "DeeperWiderAutoencoder":
            model = DeeperWiderAutoencoder(input_channels=lights)
        elif model_name == "NarrowerAutoencoder":
            model = NarrowerAutoencoder(input_channels=lights)
        elif model_name == "SparserDownSampleAutoencoder":
            model = SparserDownSampleAutoencoder(input_channels=lights, base_ch=128)
        elif model_name == "HighFreqUNetAE":
            model = HighFreqUNetAE(in_channels=lights, base_ch=128)
        elif model_name == "WiderAutoencoder":
            model = WiderAutoencoder(input_channels=lights)
        elif model_name == "ResNetAutoencoder":
            model = ResNetAutoencoder(channels=lights)

        
        modelName =f"{model_name}_{lights}"
        model_path = f"models/Trained_models/LightTest/LightTestV2/{model_name}_{lights}.pth"
        if not os.path.exists(model_path):
            print(f"Model file {model_path} does not exist. Skipping...")
            continue

        model.load_state_dict(torch.load(f"models/Trained_models/LightTest/LightTestV2/{model_name}_{lights}.pth"))
        model.eval()
        print(f"Model {model_name} loaded with {lights} lightsources")
            
        # Prepare for test
        dataset = Dataloader(dataset_path)
        train_loader, val_loader, test_loader = dataset.load_train_vali_test_dataloaders_with_n_images(n_images=lights, trainSplit=0.8, BS=1)
        if rw:
            test_loader = dataset.load_test_dataloader(n_images=lights, BS=1, path="/Test_Small_defects")


        # Calculate Threshold
        Threshold = Trainer.get_maxPixelThreshold(model=model, train_loader=train_loader)

        # Test the model and return the metrics
        Trainer.compare_images(model=model, test_loader=test_loader, threshold=Threshold, channels=lights, thresholdType="maxPixel", FigSavePath=Fig_SavePath, ModelName=modelName, display=display)

        
    

