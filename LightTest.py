import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from models.DeeperWiderAutoencoder import DeeperWiderAutoencoder

from src.Dataloader import Dataloader
from src.Trainer import Trainer

def main():
    #model.load_state_dict(torch.load("models\Trained_models\DeeperWider_autoencoder.pth"))
    #Loop that trains the model with different amount of lightsources:
    print("Loading dataset")
    dataset = Dataloader(path='Datasets/Dataset003/Train', n_lights=24, width=224, height=224, top_light=True)
    print("Dataset loaded")
    i = 0 
    thresholds = []
    accuracys = []
    precisions = []
    print("Starting test")
    while i <= 24:
        i += 4
        if i > 24:
            break
        print(f"Testing with {i} lightsources")
        #Load the data
        #TODO: This is a list of images, we need them to be loaded into a train and validation dataloader
        train_loader, test_loader = dataset.load_train_test_dataloaders_with_n_images(train_path='Datasets/Dataset003/Train', test_path='Datasets/Dataset003/Test', n_images=i)

        #Load the model
        model = DeeperWiderAutoencoder(input_channels=i)

        #Trainer.train_model(model=model, train_loader=train_loader, val_loader=test_loader, num_epochs=200, lr=0.001, save_path=f"models/Trained_models/DeeperWider_autoencoder_{i}.pth", patience=50)

        #Load the model
        model.load_state_dict(torch.load(f"models/Trained_models/DeeperWider_autoencoder_{i}.pth"))
        model.eval()
        print("Model loaded")

        # Calculate Threshold
        Threshold = Trainer.get_threshold(model=model, train_loader=train_loader)
        thresholds.append(Threshold)
        print(f"Threshold for model with {i} lightsources: {Threshold}")

        # Test the model and return the metrics
        accuracy, precision, conf_matrix, class_report = Trainer.validate(model=model, val_loader=test_loader, threshold=Threshold)
        accuracys.append(accuracy)
        precisions.append(precision)

        # Show the confusion matrix and classification report
        print(f"Confusion matrix for model with {i} lightsources:")
        print(conf_matrix)
        print(f"Classification report for model with {i} lightsources:")
        print(class_report)

        print(f"Accuracy {accuracy} and precision {precision}for model with {i} lightsources.")

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
    #plt.savefig('results/Threshold_Accuracy_Precision_vs_Number_of_Lightsources.png')
    plt.show()
    print("Test finished")


main()


        
        
