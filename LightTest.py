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
    "TruelyWiderAutoencoder",
    "HighFreqUNetAE",
]

if __name__ == "__main__":
    #dataset_path = 'Datasets/Dataset004'
    dataset_path = 'Datasets/Dataset003'
    Fig_SavePath = "LightTestFigures/"
    display = False
    
    best_model_performance = []

    for model_name in models:
        i = 0 
        thresholds = []
        accuracys = []
        precisions = []
        while i <= 24:
            i += 4
            
            if i > 24:
                break
            
            print(f"Testing {model_name} with {i} lightsources")

            # Load the dataset
            dataset = Dataloader(dataset_path)
            train_loader, val_loader, test_loader = dataset.load_train_vali_test_dataloaders_with_n_images(n_images=i, trainSplit=0.8, BS=16)
            print(f"Dataset loaded with {i} lightsources, and batch size of 16")

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
            
            print(f"Model {model_name} loaded with {i} lightsources")

            # Train the model
            modelName =f"{model_name}_{i}"
            Trainer.train_model(model=model, train_loader=train_loader, val_loader=val_loader, num_epochs=200, lr=0.001, save_path=f"models/Trained_models/LightTest/{model_name}_{i}.pth", patience=40, FigSavePath=Fig_SavePath, ModelName=modelName, display=display)

            # Prepare for test
            dataset = Dataloader(dataset_path)
            train_loader, val_loader, test_loader = dataset.load_train_vali_test_dataloaders_with_n_images(n_images=i, trainSplit=0.8, BS=1)
            print(f"Dataset loaded with {i} lightsources, and batch size of 1")

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
                model = ResNetAutoencoder(num_channels=i)

            model.eval()
            print(f"Model {model_name} loaded with {i} lightsources")

            # Calculate Threshold
            Threshold = Trainer.get_threshold2(model=model, train_loader=train_loader)
            thresholds.append(Threshold)
            print(f"Threshold for model {model_name} with {i} lightsources: {Threshold}")

            # Test the model and return the metrics
            accuracy, precision, conf_matrix, class_report = Trainer.validate(model=model, val_loader=test_loader, threshold=Threshold, FigSavePath=Fig_SavePath, ModelName=modelName, display=display)
            accuracys.append(accuracy)
            precisions.append(precision)
        
        # Plot the results
        plt.figure(figsize=(10, 5))
        plt.plot(range(4, 25, 4), thresholds, marker='o', label='Threshold')
        plt.plot(range(4, 25, 4), accuracys, marker='o', label='Accuracy')
        plt.plot(range(4, 25, 4), precisions, marker='o', label='Precision')
        plt.title('Threshold, Accuracy and Precision vs Number of Lightsources')
        plt.xlabel('Number of Lightsources')
        plt.ylabel('Value')
        plt.xticks(range(4, 25, 4))
        plt.grid()
        plt.legend()
        os.makedirs(os.path.dirname(Fig_SavePath), exist_ok=True)
        plt.savefig(f"{Fig_SavePath}{model_name}Threshold_Accuracy_Precision_vs_Number_of_Lightsources.png")
        if display:
            plt.show()
        else:
            plt.close()

        # Add the best model and its performance to the list
        best_model_performance.append({
            "model": model_name,
            "best_accuracy": max(accuracys),
            "best_precision": max(precisions),
            "best_lightsources": accuracys.index(max(accuracys)) * 4 + 4
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
    plt.ylabel('Best Accuracy')
    plt.title('Model Accuracy vs. Best Light Sources')
    plt.tight_layout()
    plt.savefig(f"{Fig_SavePath}Best_Accuracy_vs_Best_Light_Sources.png")
    if display:
        plt.show()
    else:
        plt.close()

    # Plot 2: Precision vs. Number of Light Sources
    plt.figure()
    plt.scatter(df['best_lightsources'], df['best_precision'])
    for i, txt in enumerate(df['model']):
        plt.annotate(txt, (df['best_lightsources'][i], df['best_precision'][i]))
    plt.xlabel('Best Number of Light Sources')
    plt.ylabel('Best Precision')
    plt.title('Model Precision vs. Best Light Sources')
    plt.tight_layout()
    plt.savefig(f"{Fig_SavePath}Best_Precision_vs_Best_Light_Sources.png")
    if display:
        plt.show()
    else:
        plt.close()
            


# def main():
#     #model.load_state_dict(torch.load("models\Trained_models\DeeperWider_autoencoder.pth"))
#     #Loop that trains the model with different amount of lightsources:
#     print("Loading dataset")
#     dataset = Dataloader(path='Datasets/Dataset003/Train', n_lights=24, width=224, height=224, top_light=True)
#     print("Dataset loaded")
#     i = 0 
#     thresholds = []
#     accuracys = []
#     precisions = []
#     print("Starting test")
#     while i <= 24:
#         i += 4
#         if i > 24:
#             break
#         print(f"Testing with {i} lightsources")
#         #Load the data
#         #TODO: This is a list of images, we need them to be loaded into a train and validation dataloader
#         train_loader, test_loader = dataset.load_train_test_dataloaders_with_n_images(train_path='Datasets/Dataset003/Train', test_path='Datasets/Dataset003/Test', n_images=i)

#         #Load the model
#         model = DeeperWiderAutoencoder(input_channels=i)

#         #Trainer.train_model(model=model, train_loader=train_loader, val_loader=test_loader, num_epochs=200, lr=0.001, save_path=f"models/Trained_models/DeeperWider_autoencoder_{i}.pth", patience=50)

#         #Load the model
#         model.load_state_dict(torch.load(f"models/Trained_models/DeeperWider_autoencoder_{i}.pth"))
#         model.eval()
#         print("Model loaded")

#         # Calculate Threshold
#         Threshold = Trainer.get_threshold(model=model, train_loader=train_loader)
#         thresholds.append(Threshold)
#         print(f"Threshold for model with {i} lightsources: {Threshold}")

        # # Test the model and return the metrics
        # accuracy, precision, conf_matrix, class_report = Trainer.validate(model=model, val_loader=test_loader, threshold=Threshold)
        # accuracys.append(accuracy)
        # precisions.append(precision)

        # # Show the confusion matrix and classification report
        # print(f"Confusion matrix for model with {i} lightsources:")
        # print(conf_matrix)
        # print(f"Classification report for model with {i} lightsources:")
        # print(class_report)

        # print(f"Accuracy {accuracy} and precision {precision}for model with {i} lightsources.")

    # # Plot the results
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(4, 25, 4), thresholds, marker='o', label='Threshold')
    # plt.plot(range(4, 25, 4), accuracys, marker='o', label='Accuracy')
    # plt.plot(range(4, 25, 4), precisions, marker='o', label='Precision')
    # plt.title('Threshold, Accuracy and Precision vs Number of Lightsources')
    # plt.xlabel('Number of Lightsources')
    # plt.ylabel('Value')
    # plt.xticks(range(4, 25, 4))
    # plt.grid()
    # plt.legend()
    # #plt.savefig('results/Threshold_Accuracy_Precision_vs_Number_of_Lightsources.png')
    # plt.show()
    # print("Test finished")


# main()


        
        
