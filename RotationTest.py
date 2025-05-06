import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from models.Autoencoder import Autoencoder

from src.Dataloader import Dataloader
from src.Trainer import Trainer

if __name__ == "__main__":
    i = 0
    while i < 2:
        if i == 0:
            #Load dataset2
            dataset = Dataloader('Datasets/Dataset002')
        elif i == 1:
            #Load dataset3
            dataset = Dataloader('Datasets/Dataset003')
        
        #train_loader, val_loader, test_loader = dataset.load_train_vali_test_dataloaders_with_n_images(n_images=4, trainSplit=0.8, BS=16)
        train_loader, val_loader, test_loader = dataset.load_train_vali_test_dataloaders_with_n_images(n_images=4, trainSplit=0.8, BS=1)
        
        print("Dataset loaded")

        #Load the model
        model = Autoencoder()
        print("Model loaded")

        #Trainer.train_model(model=model, train_loader=train_loader, val_loader=val_loader, num_epochs=150, lr=0.001, save_path=f"models/Trained_models/Autoencoder_{i}rotation.pth", patience=40)
        #print("Model trained")

        #Load the model
        model.load_state_dict(torch.load(f"models/Trained_models/Autoencoder_{i}rotation.pth"))
        model.eval()
        print("Model loaded")

        #Calculate Threshold
        #Threshold = Trainer.get_threshold(model=model, train_loader=train_loader)
        Threshold = Trainer.get_threshold2(model=model, train_loader=train_loader)
        print(f"Threshold for model with {i} rotation: {Threshold}")

        #Test the model and return the metrics
        accuracy, precision, conf_matrix, class_report = Trainer.validate(model=model, val_loader=test_loader, threshold=Threshold)
       
        i += 1
        print("Next dataset")
        print("======================================")