from src.Dataloader import Dataloader
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score

def calculate_average_images(images):
    """
    Calculate the average of a list of images.
    :param images: List of images (numpy arrays).
    :return: Average image (numpy array).
    """
    # Convert list of images to numpy array
    images = np.array(images)
    
    # Calculate the average image
    average_image = np.mean(images, axis=0)
    
    return average_image

def max_loss_criterion(outputs, inputs):
    # Calculate the maximum loss between the outputs and inputs
    max_loss = torch.max(torch.abs(outputs - inputs))
    return max_loss

def get_threshold(average_image, criterion, train_loader, std=3):
    MSE = []
    for inputs in train_loader:
        # inputs = inputs.cuda()
        mse = criterion(average_image, inputs).item()
        MSE.append(mse)
    mean = np.mean(MSE)
    variance = np.var(MSE)
    threshold = mean + std * np.sqrt(variance)
    return threshold

def predict_labels(average_image, criterion, test_loader, threshold):
        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                mse = criterion(average_image, inputs).item()

                y_true.extend(labels.numpy())
                y_pred.extend([1 if mse > threshold else 0] * len(labels))

        return np.array(y_true), np.array(y_pred)

def hist(average_image, test_loader, train_loader, criterion):
    """
    Plot the histogram of MSE for normal and abnormal images.
    """
    MSE_normal = []
    MSE_abnormal = []
    for inputs, labels in test_loader:
        mse = criterion(average_image, inputs).item()
        if labels == 0:
            MSE_normal.append(mse)
        else:
            MSE_abnormal.append(mse)

    for inputs in train_loader:
        mse = criterion(average_image, inputs).item()
        MSE_normal.append(mse)

    variancePlot = np.var(MSE_normal)


    plt.figure(dpi=250, figsize=(6, 4))
    plt.hist(MSE_normal, bins=25, alpha=0.5, label='Normal', edgecolor='black')
    plt.hist(MSE_abnormal, bins=25, alpha=0.5, label='Abnormal', edgecolor='black')
    plt.axvline(x=np.mean(MSE_normal), color='r', linestyle='dashed', linewidth=1, label='Mean')
    plt.axvline(x=np.mean(MSE_normal) + np.sqrt(variancePlot), color='g', linestyle='dashed', linewidth=1, label='1 $\sigma$')
    plt.axvline(x=np.mean(MSE_normal) - np.sqrt(variancePlot), color='g', linestyle='dashed', linewidth=1)
    plt.axvline(x=np.mean(MSE_normal) + 2*np.sqrt(variancePlot), color='y', linestyle='dashed', linewidth=1, label='2 $\sigma$')
    plt.axvline(x=np.mean(MSE_normal) - 2*np.sqrt(variancePlot), color='y', linestyle='dashed', linewidth=1)
    plt.axvline(x=np.mean(MSE_normal) + 3*np.sqrt(variancePlot), color='b', linestyle='dashed', linewidth=1, label='3 $\sigma$')
    plt.axvline(x=np.mean(MSE_normal) - 3*np.sqrt(variancePlot), color='b', linestyle='dashed', linewidth=1)




    plt.title('Histogram for averaging method based on statistical analysis')
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.legend()

if __name__ == "__main__":
    dataset_path = "Datasets/Dataset004"
    # Load the data
    dataset = Dataloader(dataset_path)
    train_loader, val_loader, test_loader = dataset.load_train_vali_test_dataloaders_with_n_images(n_images=4, trainSplit=0.99, BS=1)
    # test_loader = dataset.load_test_dataloader(4,1,"/Test_easy")

    print(f"Number of training images: {len(train_loader.dataset)}")
    # Extract images from train_loader as a numpy array

    images = []
    for batch in train_loader:
        images.extend(batch.detach().cpu().numpy())
    
    # Convert list of images to numpy array
    images = np.array(images)
    # Print the shape of the first image
    print(f"Shape of first image: {images.shape}")
    # Calculate the average image
    average_image = calculate_average_images(images)
    # Print the shape of the average image
    print(f"Shape of average image: {average_image.shape}")

    # Convert the average image to a tensor and move it to GPU
    average_image = torch.tensor(average_image)
    # Add a batch dimension
    average_image = average_image.unsqueeze(0)
    # Print the shape of the average image tensor
    print(f"Shape of average image tensor: {average_image.shape}")

    # Define the loss criterion
    criterion = max_loss_criterion
    # Calculate the threshold
    threshold = get_threshold(average_image, criterion, train_loader, std=3)
    # Print the threshold
    print(f"Threshold: {threshold}")

    # Plot the histogram of MSE for normal and abnormal images
    hist(average_image, test_loader, train_loader, criterion)
    plt.show()

    # Predict labels for the test set
    y_true, y_pred = predict_labels(average_image, criterion, test_loader, threshold)
    # Print the classification report
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal']))
    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)






