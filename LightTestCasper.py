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
from src.Autoencoder import Autoencoder as AlternativeAutoencoder

from src.Dataloader import Dataloader
from src.TrainerCasper import Trainer

import matplotlib as mpl
import pandas as pd

# Enable LaTeX rendering TODO: Install latex either MiKTeX or TeX Live
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'  # LaTeX default font is Computer Modern (serif)

# TODO: Train the last two models and test everyting agian

# Create a list of models to test
models = [
    AlternativeAutoencoder,
    # "Autoencoder",
    # "DeeperAutoencoder",
    # "DeeperWiderAutoencoder",
    # "NarrowerAutoencoder",
    WiderAutoencoder
    # "ResNetAutoencoder",
    # "TruelyWiderAutoencoder",
    # "HighFreqUNetAE",
]
def get_threshold(model, criterion, train_loader, std=3):
    MSE = []
    for inputs in train_loader:
        inputs = inputs.cuda()
        outputs = model(inputs)
        mse = criterion(outputs, inputs).item()
        MSE.append(mse)
    mean = np.mean(MSE)
    variance = np.var(MSE)
    threshold = mean + std * np.sqrt(variance)
    return threshold

if __name__ == "__main__":
    #dataset_path = 'Datasets/Dataset004'
    dataset_path = 'Datasets/Dataset002'
    Fig_SavePath = "LightTestFigures/"
    display = False
    
    best_model_performance = []

    for model_class in models:
        i = 0 
        thresholds = []
        accuracys = []
        precisions = []
        light_counts = [1, 2, 3, 4] # 
        # light_counts = [4]
        model_name = model_class.__name__
        print(f"Testing {model_name}")
        for i in light_counts:
            
            print(f"Testing {model_name} with {i} lightsources")

            # Load the dataset
            dataset = Dataloader(dataset_path)
            train_loader, val_loader, test_loader = dataset.load_train_vali_test_dataloaders_with_n_images(n_images=i, trainSplit=0.8, BS=16)
            print(f"Dataset loaded with {i} lightsources, and batch size of 16")

            model = model_class(i)
            model.train()
            

            # Train the model
            modelName =f"{model_name}_{i}"
            model = Trainer.train_model(model=model, train_loader=train_loader, val_loader=val_loader, num_epochs=50, lr=0.001, save_path=f"savedModels/lightTest/{model_name}_{i}.pth", patience=40, FigSavePath=Fig_SavePath, ModelName=modelName, display=display, verbose = False)

            # Prepare for test
            dataset = Dataloader(dataset_path)
            train_loader, val_loader, test_loader = dataset.load_train_vali_test_dataloaders_with_n_images(n_images=i, trainSplit=0.8, BS=1)
            print(f"Dataset loaded with {i} lightsources, and batch size of 1")


            model.eval()
            Threshold = get_threshold(model=model, criterion=torch.nn.MSELoss(), train_loader=train_loader, std = 2)
            print(f"Threshold for model {model_name} with {i} lightsources: {Threshold}")

            # Test the model and return the metrics
            accuracy, precision, conf_matrix, class_report = Trainer.validate(model=model, val_loader=test_loader, threshold=Threshold, FigSavePath=Fig_SavePath, ModelName=modelName, display=display)
            Trainer.hist(model=model, criterion=torch.nn.MSELoss(), train_loader=train_loader,test_loader=test_loader, FigSavePath=Fig_SavePath, ModelName=modelName)
            accuracys.append(accuracy)
            precisions.append(precision)
            thresholds.append(Threshold)
        
        # Plot the results
        plt.figure(figsize=(10, 5))
        plt.plot(light_counts, thresholds, marker='o', label='Threshold')
        plt.plot(light_counts, accuracys, marker='o', label='Accuracy')
        plt.plot(light_counts, precisions, marker='o', label='Precision')
        plt.title('Threshold, Accuracy and Precision vs Number of Lightsources')
        plt.xlabel('Number of Lightsources')
        plt.ylabel('Value')
        plt.xticks(light_counts)
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


        
        
