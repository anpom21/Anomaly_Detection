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

# Create a list of models to test
models = [
    "Autoencoder",
    "DeeperAutoencoder",
    "DeeperWiderAutoencoder",
    "NarrowerAutoencoder",
    "WiderAutoencoder",
    "ResNetAutoencoder",
    # "SparserDownSampleAutoencoder",
    # "HighFreqUNetAE",
]

if __name__ == "__main__":
    dataset_path = 'Datasets/IRL_4_channel_dataset'
    Fig_SavePath = "RW_Results/IRL_4_channel/"
    display = False

    model_performance = []

    for model_name in models:
        lights = 4
        # Load the dataset
        # dataset = Dataloader(dataset_path)
        # train_loader, val_loader, test_loader = dataset.load_train_vali_test_dataloaders_with_n_images(n_images=lights, trainSplit=0.9, BS=16)

        # # Load the model
        # if model_name == "Autoencoder":
        #     model = Autoencoder(input_channels=lights)
        # elif model_name == "DeeperAutoencoder":
        #     model = DeeperAutoencoder(input_channels=lights)
        # elif model_name == "DeeperWiderAutoencoder":
        #     model = DeeperWiderAutoencoder(input_channels=lights)
        # elif model_name == "NarrowerAutoencoder":
        #     model = NarrowerAutoencoder(input_channels=lights)
        # elif model_name == "SparserDownSampleAutoencoder":
        #     model = SparserDownSampleAutoencoder(input_channels=lights, base_ch=128)
        # elif model_name == "HighFreqUNetAE":
        #     model = HighFreqUNetAE(in_channels=lights, base_ch=128)
        # elif model_name == "WiderAutoencoder":
        #     model = WiderAutoencoder(input_channels=lights)
        # elif model_name == "ResNetAutoencoder":
        #     model = ResNetAutoencoder(channels=lights)

        # Train the model
        modelName =f"{model_name}RW_{lights}"
        #Trainer.train_model(model=model, train_loader=train_loader, val_loader=val_loader, num_epochs=200, lr=0.001, save_path=f"models/Trained_models/RW_Test/IRL_4_channel/{model_name}_{i}.pth", patience=40, FigSavePath=Fig_SavePath, ModelName=modelName, display=display)

        # Prepare for test
        dataset = Dataloader(dataset_path)
        train_loader, val_loader, test_loader = dataset.load_train_vali_test_dataloaders_with_n_images(n_images=lights, trainSplit=0.8, BS=1)
        test_loader = dataset.load_test_dataloader(n_images=lights, BS=1, path="/Test_Small_defects")

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

        model.load_state_dict(torch.load(f"models/Trained_models/RW_Test/IRL_4_channel/{model_name}_{lights}.pth"))
        model.eval()
        print(f"Model {model_name} loaded with {lights} lightsources")

        # Calculate Threshold
        Threshold = Trainer.get_maxPixelThreshold(model=model, train_loader=train_loader)
        print(f"Threshold for model {model_name} with {lights} lightsources: {Threshold}")

        # Test the model and return the metrics
        accuracy, precision, conf_matrix, class_report, ROCThresholds, roc_auc = Trainer.validate(model=model, val_loader=test_loader, threshold=Threshold, thresholdType="maxPix", FigSavePath=Fig_SavePath, ModelName=modelName, display=display)

        # Save the metrics
        model_performance.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
        })

        print(f"Testing {model_name} finished")
    
    # Plot the results
    df = pd.DataFrame(model_performance)

    plt.figure(figsize=(10, 6))
    plt.scatter(df["Model"], df["Accuracy"], color='blue', alpha=0.7, label='Accuracy')
    #plt.bar(df["Model"], df["Precision"], color='orange', alpha=0.7, label='Precision')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title(f'Real world data Model Performance on {lights} lightsources')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(Fig_SavePath, "Model_Performance.png"), dpi=300)
    plt.show()
    print("Model performance saved as Model_Performance.png")

    # Print the name of the models and their accuracy
    for model in model_performance:
        print(f"Model: {model['Model']}, Accuracy: {model['Accuracy']:.4f}")