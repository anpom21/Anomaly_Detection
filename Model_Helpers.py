import tqdm
import torch
import matplotlib.pyplot as plt
import HelperFunctions as hf
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score

# function to train the model
def train_model(model, train_loader, test_loader, save_path='simple_autoencoder2_l2_loss.pth'):
    #model = Autoencoder() 
    #model.cuda()# Move the model to the GPU
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

            # smooth the image using Gaussian filter
            #img = torchvision.transforms.functional.gaussian_blur(img, kernel_size=3, sigma=(0.1, 2.0))
            
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
    #torch.save(model.state_dict(), 'simple_autoencoder2_l2_loss.pth')
    torch.save(model.state_dict(), save_path)
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
    data = hf.load_np_data(False)

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
            test_image = hf.transform(Image.open(path)).cuda().unsqueeze(0)
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

# process the test dataset
def test(dataset, model):
    #load images from the dataset
    data = torch.stack([img for img, _ in dataset])

    with torch.no_grad():
        data = data.cuda()
        recon = model(data)
        
    recon_error =  ((data-recon)**2).mean(axis=1)
    print(recon_error.shape)
        
    for i in range(98):
        #save the error image
        error_array = recon_error[i][0:-10,0:-10].cpu().numpy()
        # Normalize to [0,255] range for proper image saving
        error_array = (error_array - error_array.min()) / (error_array.max() - error_array.min())  # Normalize to [0,1]
        error_array = (error_array * 255).astype(np.uint8)  # Scale to [0,255] and convert to uint8
        error_img = Image.fromarray(error_array, mode="L")  # Convert to grayscale
        error_img.save(f"DeeperWider/error_{i}.png")
        if(i % 10 == 0):
            print(f"max error in image {i}: ", torch.max(recon_error[i]))
            #Threshold the error image to only show score above 0.01
            error_img = np.where(recon_error[i][0:-10,0:-10].cpu().numpy() > 0.01, recon_error[i][0:-10,0:-10].cpu().numpy(), 0)
            #Display the error image and the original image
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(data[i].cpu().numpy().transpose((1, 2, 0)))
            plt.title(f'Original Image {i}')
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(recon[i].cpu().numpy().transpose((1, 2, 0)))
            plt.title(f'Reconstructed Image {i}')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(error_img, cmap='jet')
            plt.title(f'Error Image {i}')
            plt.axis('off')
            plt.show()
