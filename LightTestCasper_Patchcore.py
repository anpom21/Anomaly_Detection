import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import os

from tqdm.auto import tqdm

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score

from src.Dataloader import Dataloader
from torchvision.models import resnet50, ResNet50_Weights

import matplotlib as mpl
import pandas as pd

# Enable LaTeX rendering TODO: Install latex either MiKTeX or TeX Live
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'  # LaTeX default font is Computer Modern (serif)

# TODO: Train the last two models and test everyting agian


def max_loss_criterion(outputs, inputs):
    # Calculate the maximum loss between the outputs and inputs
    max_loss = torch.max(torch.abs(outputs - inputs))
    return max_loss

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

class resnet_feature_extractor(torch.nn.Module):
    def __init__(self, input_channels= 4):
        """This class extracts the feature maps from a pretrained Resnet model."""
        super(resnet_feature_extractor, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Average the weights of the first convolution but retain 3 input layers
        weights = self.model.conv1.weight.clone()
        weights_avg = torch.mean(weights, dim=1, keepdim=True)
    
        weights = torch.cat([weights_avg for _ in range(input_channels)], dim=1)
        self.model.conv1.weight = torch.nn.Parameter(weights)
        self.model.conv1.in_channels = input_channels
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        

        # Hook to extract feature maps
        def hook(module, input, output) -> None:
            """This hook saves the extracted feature map on self.featured."""
            self.features.append(output)

        self.model.layer2[-1].register_forward_hook(hook)            
        self.model.layer3[-1].register_forward_hook(hook) 

    def forward(self, input):

        self.features = []
        with torch.no_grad():
            _ = self.model(input)

        self.avg = torch.nn.AvgPool2d(3, stride=1)
        fmap_size = self.features[0].shape[-2]         # Feature map sizes h, w
        self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)

        resized_maps = [self.resize(self.avg(fmap)) for fmap in self.features]
        patch = torch.cat(resized_maps, 1)            # Merge the resized feature maps
        patch = patch.reshape(patch.shape[1], -1).T   # Craete a column tensor

        return patch

if __name__ == "__main__":
    
    dataset_path = 'Datasets/Dataset004'
    Fig_SavePath = "LightTestFigures/"
    display = False
    

    i = 0 
    thresholds = []
    accuracys = []
    precisions = []
    light_counts = range(1,24)#[1, 2, 3, 4, 8, 12, 16, 20, 24] # 
    # light_counts = [4]
    for i in light_counts:
        
        # Load the dataset
        dataset = Dataloader(dataset_path)
        train_loader, val_loader, test_loader = dataset.load_train_vali_test_dataloaders_with_n_images(n_images=i, trainSplit=0.8, BS=1)
        print(f"{i} lightsources loaded")

        backbone = resnet_feature_extractor(input_channels = i)
        backbone = backbone.cuda()
        # ---------------------------- Create memory bank ---------------------------- #
        memory_bank =[]


        for image in tqdm(train_loader.dataset):

            with torch.no_grad():
                data = image.unsqueeze(0).cuda()
                features = backbone(data)
                memory_bank.append(features.cpu().detach())

        memory_bank = torch.cat(memory_bank,dim=0).cuda()

        # ----------------------------- Calculate threshold ----------------------------- #
        y_score=[]

        for image in tqdm(train_loader.dataset):
            data = image.unsqueeze(0).cuda()
            with torch.no_grad():
                features = backbone(data)
            distances = torch.cdist(features, memory_bank, p=2.0)
            dist_score, dist_score_idxs = torch.min(distances, dim=1) 
            s_star = torch.max(dist_score)
            segm_map = dist_score.view(1, 1, 28, 28) 

            y_score.append(s_star.cpu().numpy())

        best_threshold = np.mean(y_score) + 3 * np.std(y_score)

        # -------------------------- Calculate on test data -------------------------- #
        y_test_score = []
        y_true=[]

        for image, label in test_loader:
            with torch.no_grad():
                test_image = image.cuda()
                features = backbone(test_image)

            distances = torch.cdist(features, memory_bank, p=2.0)
            dist_score, dist_score_idxs = torch.min(distances, dim=1) 
            s_star = torch.max(dist_score)
            segm_map = dist_score.view(1, 1, 28, 28) 

            y_test_score.append(s_star.cpu().numpy())
            y_true.append(label)

        y_guess = np.array(y_test_score) > best_threshold
        accuracy = accuracy_score(y_true, y_guess)
        precision = precision_score(y_true, y_guess, zero_division=0)
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")

        thresholds.append(best_threshold)
        accuracys.append(accuracy)
        precisions.append(precision)

        conf_matrix = confusion_matrix(y_true, y_guess)
        class_report = classification_report(y_true, y_guess)

    # Plot the results
    plt.figure(figsize=(10, 5))
    print(light_counts)
    print(f"Thresholds: {thresholds}")
    print(f"Accuracys: {accuracys}")
    print(f"Precisions: {precisions}")
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
    plt.savefig(f"{Fig_SavePath}PatchCore_Threshold_Accuracy_Precision_vs_Number_of_Lightsources.png")
    if display:
        plt.show()
    else:
        plt.close()

        
        
