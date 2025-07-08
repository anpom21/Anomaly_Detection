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

light_sources = [1, 2, 3, 4, 8, 12, 16, 20, 24]  # Define the number of light sources to test
#light_sources = [16,17,18,19,20,21,22,23,24]  # Define the number of light sources to test

if __name__ == "__main__":
    #dataset_path = 'Datasets/Dataset004'
    dataset_path = 'Datasets/Dataset004'
    Fig_SavePath = "LightTestFigures/LightTestV2/ResNetAutoencoder/"
    display = False
    
    best_model_performance = []

    model_performance = []

    for model_name in models:
        thresholds = []
        accuracys = []
        precisions = []
        valid_light_sources = []
        for lights in light_sources:

            print(f"Testing {model_name} with {lights} lightsources")

            # Load the dataset
            dataset = Dataloader(dataset_path)
            train_loader, val_loader, test_loader = dataset.load_train_vali_test_dataloaders_with_n_images(n_images=lights, trainSplit=0.8, BS=16)
            print(f"Dataset loaded with {lights} lightsources, and batch size of 16")

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
            
            print(f"Model {model_name} loaded with {lights} lightsources")

            # Train the model
            modelName =f"{model_name}_{lights}"
            model_path = f"models/Trained_models/LightTest/LightTestV2/{model_name}_{lights}.pth"
            #if not os.path.exists(model_path):
                # Trainer.train_model(model=model, train_loader=train_loader, val_loader=val_loader, num_epochs=200, lr=0.001, save_path=f"models/Trained_models/LightTest/LightTestV2/{model_name}_{lights}.pth", patience=40, FigSavePath=Fig_SavePath, ModelName=modelName, display=display)

            # Prepare for test
            dataset = Dataloader(dataset_path)
            train_loader, val_loader, test_loader = dataset.load_train_vali_test_dataloaders_with_n_images(n_images=lights, trainSplit=0.8, BS=1)
            print(f"Dataset loaded with {lights} lightsources, and batch size of 1")

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

            model_path = f"models/Trained_models/LightTest/LightTestV2/{model_name}_{lights}.pth"
            if not os.path.exists(model_path):
                print(f"Model file {model_path} does not exist. Skipping...")
                continue

            model.load_state_dict(torch.load(f"models/Trained_models/LightTest/LightTestV2/{model_name}_{lights}.pth"))
            model.eval()
            valid_light_sources.append(lights)
            print(f"Model {model_name} loaded with {lights} lightsources")

            # Calculate Threshold
            Threshold = Trainer.get_maxPixelThreshold(model=model, train_loader=train_loader)
            thresholds.append(Threshold)
            print(f"Threshold for model {model_name} with {lights} lightsources: {Threshold}")

            # Test the model and return the metrics
            accuracy, precision, conf_matrix, class_report, ROCThresholds, roc_auc = Trainer.validate(model=model, val_loader=test_loader, threshold=Threshold, thresholdType="maxPix", FigSavePath=Fig_SavePath, ModelName=modelName, display=display)
            accuracys.append(accuracy)
            precisions.append(precision)
            
            model_performance.append({
                "model": model_name,
                "lightsources": lights,
                "accuracy": accuracy,
            })
        
        # Plot the results
        plt.figure(figsize=(10, 5))
        plt.plot(valid_light_sources, accuracys, marker='o', label='Accuracy')
        plt.title('Accuracy vs Number of Lightsources')
        plt.xlabel('Number of Lightsources')
        plt.ylabel('Value')
        plt.xticks(light_sources)
        plt.grid()
        plt.legend()
        os.makedirs(os.path.dirname(Fig_SavePath), exist_ok=True)
        plt.savefig(f"{Fig_SavePath}{model_name}Accuracy_vs_Number_of_Lightsources.png")
        if display:
            plt.show()
        else:
            plt.close()

        # Add the best model and its performance to the list
        best_index = accuracys.index(max(accuracys))
        best_model_performance.append({
            "model": model_name,
            "best_accuracy": max(accuracys),
            "best_precision": max(precisions),
            "best_lightsources": valid_light_sources[best_index]
        })

        print(f"Testing {model_name} finished")
    
    # Plot a graph of the best model performances
    # Convert to DataFrame
    df = pd.DataFrame(best_model_performance)

    # Plot 1: Accuracy vs. Number of Light Sources
    plt.figure()
    plt.scatter(df['best_lightsources'], df['best_accuracy'])
    for i, txt in enumerate(df['model']):
        plt.annotate(txt, (df['best_lightsources'][i], df['best_accuracy'][i]))
    plt.xlabel('Best Number of Light Sources')
    plt.xticks(light_sources)
    plt.ylabel('Best Accuracy')
    plt.title('Model Accuracy vs. Best Light Sources')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{Fig_SavePath}Best_Accuracy_vs_Best_Light_Sources.png")
    if display:
        plt.show()
    else:
        plt.close()

    # Plot 2: All models accuracy vs. number of light sources
    model_dict = {}
    for model in model_performance:
        if model['model'] not in model_dict:
            model_dict[model['model']] = {'lightsources': [], 'accuracy': []}
        model_dict[model['model']]['lightsources'].append(model['lightsources'])
        model_dict[model['model']]['accuracy'].append(model['accuracy'])
    
    
    plt.figure(figsize=(10, 6))
    for model_name, performance in model_dict.items():
        plt.plot(performance['lightsources'], performance['accuracy'], marker='o', label=model_name)
    plt.title('All Model Performance vs Number of Light Sources')
    plt.xlabel('Number of Light Sources')
    plt.ylabel('Accuracy')
    plt.xticks(light_sources)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{Fig_SavePath}All_Model_Performance_vs_Number_of_Light_Sources.png")
    if display:
        plt.show()
    else:
        plt.close()

            
