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

# TODO: figure out why normal histograms becomes small and tall

# Create a list of models to test
models = [
    "Autoencoder",
]

if __name__ == "__main__":
    dataset_path = 'Datasets/Dataset002'
    Fig_SavePath = "ThresholdTest/Dataset002/"
    display = True

    
    for model_name in models:
        print(f"Testing {model_name}")
        
        # Load the dataset
        dataset = Dataloader(dataset_path)
        train_loader, val_loader, test_loader = dataset.load_train_vali_test_dataloaders_with_n_images(n_images=4, trainSplit=0.8, BS=16)
        print(f"Dataset loaded with 4 lightsources, and batch size of 16")

        # Load the model
        if model_name == "Autoencoder":
            model = Autoencoder(input_channels=4)
            
        print(f"Model {model_name} loaded with 4 lightsources")

        # Train the model
        modelName =f"{model_name}_thresholdTest002"
        #Trainer.train_model(model=model, train_loader=train_loader, val_loader=val_loader, num_epochs=200, lr=0.001, save_path=f"models/Trained_models/LightTest/{modelName}.pth", patience=40, FigSavePath=Fig_SavePath, ModelName=modelName, display=display)
            

        # Prepare for test
        dataset = Dataloader(dataset_path)
        train_loader, val_loader, test_loader = dataset.load_train_vali_test_dataloaders_with_n_images(n_images=4, trainSplit=0.8, BS=1)
        print(f"Dataset loaded with 4 lightsources, and batch size of 1")

        # Load the model
        if model_name == "Autoencoder":
            model = Autoencoder(input_channels=4)

        model.load_state_dict(torch.load(f"models/Trained_models/LightTest/{modelName}.pth"))
        model.eval()
        print(f"Model {model_name} loaded with 4 lightsources")

        # Calculate Thresholds
        thresholdMaxPix = Trainer.get_maxPixelThreshold(model=model, train_loader=train_loader)
        thresholdMaxPixMSE = Trainer.get_maxPixelMSEThreshold(model=model, train_loader=train_loader)
        thresholdMSE = Trainer.get_MSEThreshold(model=model, train_loader=train_loader)
        
        # Test the model  with the different thresholds and return the metrics
        accuracyMaxPix, precisionMaxPix, conf_matrixMaxPix, class_reportMaxPix, ROCThresholdsMaxPix, roc_aucMaxPix = Trainer.validate(model=model, val_loader=test_loader, threshold=thresholdMaxPix, thresholdType="maxPix", FigSavePath=Fig_SavePath, ModelName=modelName, display=display)
        accuracyMaxPixMSE, precisionMaxPixMSE, conf_matrixMaxPixMSE, class_reportMaxPixMSE, ROCThresholdsMaxPixMSE, roc_aucMaxPixMSE = Trainer.validate(model=model, val_loader=test_loader, threshold=thresholdMaxPixMSE, thresholdType="maxPixMSE", FigSavePath=Fig_SavePath, ModelName=modelName, display=display)
        accuracyMSE, precisionMSE, conf_matrixMSE, class_reportMSE, ROCThresholdsMSE, roc_aucMSE = Trainer.validate(model=model, val_loader=test_loader, threshold=thresholdMSE, thresholdType="MSE", FigSavePath=Fig_SavePath, ModelName=modelName, display=display)

        print(f"Testing {model_name} finished")
            
