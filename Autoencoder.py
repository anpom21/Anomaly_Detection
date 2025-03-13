import torch
import torch.nn as nn
import tqdm
from pathlib import Path
import numpy as np
import os, shutil
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchsummary import summary
from torch.utils.data.dataset import Subset
import urllib.request
import tarfile
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score
import seaborn as sns

# Define autoencoder anomaly detection model class
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=4),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, output_padding=1 ),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 4, kernel_size=5, stride=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Create a custom dataset with images and empty labels
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0  # Ensure float32
        return torch.tensor(img), self.labels[idx]
        #return self.images[idx], self.labels[idx]  # Return (image, empty list)

# Define the transformation pipeline using torchvision.transforms.Compose
transform = transforms.Compose([
    transforms.Resize((224,224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor()          # Convert the image to a PyTorch tensor and divide by 255.0
])

transform_np = transforms.Compose([
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor and divide by 255.0
    transforms.Resize((224,224))  # Resize the image to 224x224 pixels 
])

# load the data and split it into training and testing sets
def load_data():
    # Download and extract the MVTec AD dataset
    if not os.path.exists('carpet'):
        urllib.request.urlretrieve("https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz", "carpet.tar.xz")

        with tarfile.open('carpet.tar.xz') as f:
            f.extractall('.')

    # Links to other objects from MVTEC AD dataset
    # "bottle": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz",
    # "cable": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz",
    # "capsule": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz",
    # "carpet": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz",
    # "grid": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz",
    # "hazelnut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz",
    # "leather": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz",
    # "metal_nut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz",
    # "pill": "https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz",
    # "screw": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz",
    # "tile": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz",
    # "toothbrush": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz",
    # "transistor": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz",
    # "wood": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz",
    # "zipper": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz"

    # Define the path to the directory containing the train images
    train_image_path = 'carpet/train'

    # Load the train dataset using the ImageFolder dataset class
    # ImageFolder is a PyTorch dataset class for loading images from a directory
    # It automatically loads images from subdirectories and applies transformations to them
    # In this case, 'transform' is a transformation applied to each image in the dataset
    # It preprocesses the images before they are used for training
    good_dataset = ImageFolder(root=train_image_path, transform=transform)

    # Access a sample from the dataset
    # In this case, we're accessing the first sample
    # x contains the preprocessed image data
    # y contains the corresponding label (class index)
    x, y = good_dataset[0]

    # Print the shape of the preprocessed image data (x) and its corresponding label (y)
    print("Image Shape:", x.shape)
    print("Label:", y)

    # Split the dataset into training and testing subsets
    # The `torch.utils.data.random_split` function randomly splits a dataset into non-overlapping subsets
    # The first argument `good_dataset` is the dataset to be split
    # The second argument `[0.8, 0.2]` specifies the sizes of the subsets. Here, 80% for training and 20% for testing.
    train_dataset, test_dataset = torch.utils.data.random_split(good_dataset, [0.8, 0.2])

    # Print the lengths of the original dataset, training subset, and testing subset
    print("Total number of samples in the original dataset:", len(good_dataset))
    print("Number of samples in the training subset:", len(train_dataset))
    print("Number of samples in the testing subset:", len(test_dataset))

    # Assuming train_dataset and test_dataset are PyTorch datasets containing image data and labels
    # Set the batch size
    BS = 16

    # Create data loaders for training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=True)

    # Get a batch of images and labels from the training loader
    image_batch, label_batch = next(iter(train_loader))

    # Print the shape of the input images and labels
    print(f'Shape of input images: {image_batch.shape}')
    print(f'Shape of labels: {label_batch.shape}')

    # Set the figure size
    plt.figure(figsize=(12*4, 48*4))

    # Create a grid of images from the image batch and visualize it
    grid = torchvision.utils.make_grid(image_batch[0:4], padding=5, nrow=4)
    plt.imshow(grid.permute(1, 2, 0))  # Permute dimensions to (height, width, channels) for visualization
    plt.title('Good Samples')  # Set the title of the plot
    plt.show()  # Show the plot

    return train_loader, test_loader

def load_np_data(train = True):
    # load the list containing the four channel images
    if train:
        data = np.load('carpet/layered_images.npy')
    else:
        data = np.load('carpet/test_layered_images.npy')

    print("Data array shape: ", data.shape)
    print("First element of data array: ", data[0].shape)

    # Load the list using the Imagefolder dataset class
    for img in data:
        #convert np.array to PIL image
        #img = Image.fromarray(img)
        img = np.moveaxis(img, 0, -1)  # Convert (C, H, W) -> (H, W, C)
        img = transform_np(img)
        print("img shape: ", img.shape, " ", img.dtype)
    
    # Create dummy labels (empty lists)
    labels = [[0] for _ in range(len(data))]  # Each image gets an empty list as a label

    dataset = CustomDataset(data, labels)

    if train != True:
        # load the first three images from the dataset into a tensor
        test = torch.stack([dataset[0][0], dataset[1][0], dataset[2][0]])
        return test

    # Split the dataset into training and testing subsets
    # The `torch.utils.data.random_split` function randomly splits a dataset into non-overlapping subsets
    # The first argument `good_dataset` is the dataset to be split
    # The second argument `[0.8, 0.2]` specifies the sizes of the subsets. Here, 80% for training and 20% for testing.
    #train_dataset, test_dataset = torch.utils.data.random_split(good_dataset, [0.75, 0.25])
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    
    # Print the lengths of the original dataset, training subset, and testing subset
    print("Total number of samples in the original dataset:", len(dataset))
    print("Number of samples in the training subset:", len(train_dataset))
    print("Number of samples in the testing subset:", len(test_dataset))

    # Assuming train_dataset and test_dataset are PyTorch datasets containing image data and labels
    # Set the batch size
    BS = 5#16

    # Create data loaders for training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=True)

    # Get a batch of images and labels from the training loader
    image_batch, label_batch = next(iter(train_loader))
    #image_batch = next(iter(train_loader))

    # Print the shape of the input images and labels
    print(f'Shape of input images: {image_batch.shape}')
    #print(f'Shape of labels: {label_batch.shape}')
    #print(f'Labels: {label_batch}')

    # Set the figure size
    plt.figure(figsize=(12*4, 48*4))

    # Create a grid of images from the image batch and visualize it
    grid = torchvision.utils.make_grid(image_batch[0:4], padding=5, nrow=4)
    plt.imshow(grid.permute(1, 2, 0))  # Permute dimensions to (height, width, channels) for visualization
    plt.title('Good Samples')  # Set the title of the plot
    plt.show()  # Show the plot

    return train_loader, test_loader


# function to train the model
def train_model(train_loader, test_loader):
    model = Autoencoder() 
    model.cuda()# Move the model to the GPU
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)

    # Define a list to store training loss and validation loss
    Loss = []
    Validation_Loss = []


    num_epochs = 200
    for epoch in tqdm(range(num_epochs)):
        model.train()  # Set model to training mode
        for img, _ in train_loader:
            img = img.cuda()
            #print("shape", img.shape, " ", img.dtype)
            
            output = model(img)
            loss = criterion(output, img)

            optimizer.zero_grad() #clears the gradients of all optimized tensors.  This step is necessary because gradients are accumulated by default in PyTorch, and we want to compute fresh gradients for the current batch of data.
            loss.backward() # This line computes the gradients of the loss function with respect to the model parameters. These gradients are used to update the model parameters during optimization.
            optimizer.step() # This line updates the model parameters using the computed gradients. 
        Loss.append(loss.item())
        

        # Calculate validation loss
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            val_loss_sum = 0.0
            num_batches = 0
            for img, _ in test_loader:
                img = img.cuda()
                output = model(img)
                val_loss = criterion(output, img)
                val_loss_sum += val_loss.item()
                num_batches += 1
            val_loss_avg = val_loss_sum / num_batches
            Validation_Loss.append(val_loss_avg)
        
        if epoch % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item(), val_loss_avg))

    plt.plot(Loss, label='Training Loss')
    plt.plot(Validation_Loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save the model
    torch.save(model.state_dict(), 'simple_autoencoder_l2_loss.pth')
    model.eval()


def validate(train_loader, model):
    with torch.no_grad():
        for data, _ in train_loader:
            data = data.cuda()
            recon = model(data)
            break

    recon_error =  ((data-recon)**2).mean(axis=1)
    print(recon_error.shape)

    plt.figure(dpi=250)
    fig, ax = plt.subplots(3, 3, figsize=(5*4, 4*4))
    for i in range(3):
        ax[0, i].imshow(data[i].cpu().numpy().transpose((1, 2, 0)))
        ax[1, i].imshow(recon[i].cpu().numpy().transpose((1, 2, 0)))
        ax[2, i].imshow(recon_error[i][0:-10,0:-10].cpu().numpy(), cmap='jet',vmax= torch.max(recon_error[i])) #[0:-10,0:-10]
        ax[0, i].axis('OFF')
        ax[1, i].axis('OFF')
        ax[2, i].axis('OFF')
    plt.show()

    
    #test_image_1 = transform(Image.open(r'.\carpet\test\color\000.png'))
    #test_image_2 = transform(Image.open(r'.\carpet\test\cut\000.png'))
    #test_image_3 = transform(Image.open(r'.\carpet\test\hole\000.png'))

    #data = torch.stack([test_image_1,test_image_2, test_image_3])
    data = load_np_data(False)

    with torch.no_grad():
        data = data.cuda()
        recon = model(data)
        
    recon_error =  ((data-recon)**2).mean(axis=1)
    print(recon_error.shape)
        
    plt.figure(dpi=250)
    fig, ax = plt.subplots(3, 3, figsize=(5*4, 4*4))
    for i in range(3):
        ax[0, i].imshow(data[i].cpu().numpy().transpose((1, 2, 0)))
        ax[1, i].imshow(recon[i].cpu().numpy().transpose((1, 2, 0)))
        ax[2, i].imshow(recon_error[i][0:-10,0:-10].cpu().numpy(), cmap='jet',vmax= torch.max(recon_error[i]))
        #save the error image
        error_array = recon_error[i][0:-10,0:-10].cpu().numpy()
        # Normalize to [0,255] range for proper image saving
        error_array = (error_array - error_array.min()) / (error_array.max() - error_array.min())  # Normalize to [0,1]
        error_array = (error_array * 255).astype(np.uint8)  # Scale to [0,255] and convert to uint8
        error_img = Image.fromarray(error_array, mode="L")  # Convert to grayscale
        error_img.save(f"error_{i}.png")
        #recon_error_img = Image.fromarray(recon_error[i][0:-10,0:-10].cpu().numpy())
        #recon_error_img.save(f"recon_error_{i}.png")
        ax[0, i].axis('OFF')
        ax[1, i].axis('OFF')
        ax[2, i].axis('OFF')
    plt.show()

    RECON_ERROR=[]
    with torch.no_grad():
        for data, _ in train_loader:
            data = data.cuda()
            recon = model(data)
            data_recon_squared_mean =  ((data-recon)**2).mean(axis=(1))[:,0:-10,0:-10].mean(axis=(1,2))
            
            RECON_ERROR.append(data_recon_squared_mean)
            
    RECON_ERROR = torch.cat(RECON_ERROR).cpu().numpy()

    best_threshold = np.mean(RECON_ERROR) + 3 * np.std(RECON_ERROR)

    plt.hist(RECON_ERROR,bins=50)
    plt.vlines(x=best_threshold,ymin=0,ymax=30,color='r') 
    plt.show()

    y_true=[]
    y_pred=[]
    y_score=[]

    model.eval()

    with torch.no_grad():

        test_path = Path('carpet/test')

        for path in test_path.glob('*/*.png'):
            fault_type = path.parts[-2]
            # if fault_type != 'good':
            test_image = transform(Image.open(path)).cuda().unsqueeze(0)
            recon_image = model(test_image)
            
            # y_score_image = 
            y_score_image =  ((test_image - recon_image)**2).mean(axis=(1))[:,0:-10,0:-10].mean()
        
            y_pred_image = 1*(y_score_image >= best_threshold)
            
            y_true_image = 0 if fault_type == 'good' else 1
            
            y_true.append(y_true_image)
            y_pred.append(y_pred_image.cpu())
            y_score.append(y_score_image.cpu())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_score = np.array(y_score)

    plt.hist(y_score,bins=50)
    plt.vlines(x=best_threshold,ymin=0,ymax=30,color='r')
    plt.show()

    # Calculate AUC-ROC score
    auc_roc_score = roc_auc_score(y_true, y_score)
    print("AUC-ROC Score:", auc_roc_score)

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_roc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()



#---------------------------------------------------------------------------------------------------------------------
# main

# Load dataset
#train_loader, test_loader = load_data()
train_loader, test_loader = load_np_data()
# Train the model
#train_model(train_loader, test_loader)

# Load the trained model
model = Autoencoder()
model.load_state_dict(torch.load('simple_autoencoder_l2_loss.pth'))
model.eval()
model.cuda()

# Validate the model
validate(train_loader, model)