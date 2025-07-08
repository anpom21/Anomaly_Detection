import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

class Trainer:
    """
    Goal of this file: contain the training loop.
    """
    def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, save_path='best_model.pth', patience=5, FigSavePath=None, ModelName=None, display=True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model = model.to(device)

        criterion = nn.MSELoss()  # Reconstruction loss for autoencoders
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_loss = float('inf')
        epochs_without_improvement = 0

        # Lists to store losses for plotting
        train_losses = []
        val_losses = []

        for epoch in tqdm(range(num_epochs)):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # === Training ===
            model.train()
            running_loss = 0.0
            total = 0

            for images in train_loader:
                if isinstance(images, (list, tuple)):
                    images = images[0]  # In case it's a tuple like (images, dummy_label)

                images = images.to(device).float()

                optimizer.zero_grad()
                outputs = model(images)

                loss = criterion(outputs, images)  # reconstruct input
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                total += images.size(0)

            train_loss = running_loss / total
            train_losses.append(train_loss)  # Store training loss
            print(f"Train Loss: {train_loss:.4f}")

            # === Validation ===
            model.eval()
            val_loss = 0.0
            val_total = 0

            with torch.no_grad():
                for images in val_loader:
                    if isinstance(images, (list, tuple)):
                        images = images[0]

                    images = images.to(device).float()
                    outputs = model(images)
                    loss = criterion(outputs, images)

                    val_loss += loss.item() * images.size(0)
                    val_total += images.size(0)

            val_loss /= val_total
            val_losses.append(val_loss)  # Store validation loss
            print(f"Val Loss: {val_loss:.4f}")

            # === Check for improvement ===
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), save_path)
                print(f"✅ New best model saved at epoch {epoch+1}")
            else:
                epochs_without_improvement += 1
                print(f"⚠️ No improvement for {epochs_without_improvement} epoch(s)")

            # === Early stopping ===
            if epochs_without_improvement >= patience:
                print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs. Best Val Loss: {best_loss:.4f}")
                break

        print(f"\n🎯 Training complete. Best validation loss: {best_loss:.4f}")

        # === Plot Loss Graph ===
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid()
        if FigSavePath is not None:
            os.makedirs(os.path.dirname(FigSavePath), exist_ok=True)
            plt.savefig(f"{FigSavePath}{ModelName}TrainValLoss.png")
        if display:
            plt.show()
        else:
            plt.close()
        

    # def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, save_path='best_model.pth', patience=5):
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     print(f"Using device: {device}")
    #     model = model.to(device)

    #     criterion = nn.MSELoss()  # Reconstruction loss for autoencoders
    #     optimizer = optim.Adam(model.parameters(), lr=lr)

    #     best_loss = float('inf')
    #     epochs_without_improvement = 0

    #     for epoch in tqdm(range(num_epochs)):
    #         print(f"\nEpoch {epoch + 1}/{num_epochs}")

    #         # === Training ===
    #         model.train()
    #         running_loss = 0.0
    #         total = 0

    #         for images in train_loader:
    #             if isinstance(images, (list, tuple)):
    #                 images = images[0]  # In case it's a tuple like (images, dummy_label)

    #             images = images.to(device).float()

    #             optimizer.zero_grad()
    #             outputs = model(images)

    #             loss = criterion(outputs, images)  # reconstruct input
    #             loss.backward()
    #             optimizer.step()

    #             running_loss += loss.item() * images.size(0)
    #             total += images.size(0)

    #         train_loss = running_loss / total
            
    #         print(f"Train Loss: {train_loss:.4f}")

    #         # === Validation ===
    #         model.eval()
    #         val_loss = 0.0
    #         val_total = 0

    #         with torch.no_grad():
    #             for images in val_loader:
    #                 if isinstance(images, (list, tuple)):
    #                     images = images[0]

    #                 images = images.to(device).float()
    #                 outputs = model(images)
    #                 loss = criterion(outputs, images)

    #                 val_loss += loss.item() * images.size(0)
    #                 val_total += images.size(0)

    #         val_loss /= val_total
    #         print(f"Val Loss: {val_loss:.4f}")

    #         # === Check for improvement ===
    #         if val_loss < best_loss:
    #             best_loss = val_loss
    #             epochs_without_improvement = 0
    #             torch.save(model.state_dict(), save_path)
    #             print(f"✅ New best model saved at epoch {epoch+1}")
    #         else:
    #             epochs_without_improvement += 1
    #             print(f"⚠️ No improvement for {epochs_without_improvement} epoch(s)")

    #         # === Early stopping ===
    #         if epochs_without_improvement >= patience:
    #             print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs. Best Val Loss: {best_loss:.4f}")
    #             break

    #     print(f"\n🎯 Training complete. Best validation loss: {best_loss:.4f}")

    # def validate(model, val_loader, threshold):
    #     """
    #     Validate the model on the validation dataset.
    #     """
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model = model.to(device)
    #     model.eval()

    #     true_labels = []
    #     predicted_labels = []

    #     with torch.no_grad():
    #         for images, labels in val_loader:  # Assuming val_loader returns (images, labels)
    #             images = images.to(device).float()
    #             labels = labels.cpu().numpy()  # Convert labels to numpy for metrics
    #             true_labels.extend(labels)

    #             # Forward pass
    #             outputs = model(images)
    #             reconstruction_error = ((images - outputs) ** 2).mean(axis=(1))  # Per-image anomaly score

    #             # Predict based on threshold
    #             for i in range(len(reconstruction_error)):
    #                 thresholded = np.where(reconstruction_error[i][0:-10,0:-10].cpu().numpy() > 0.01, reconstruction_error[i][0:-10,0:-10].cpu().numpy(), 0)
    #                 predicted = 1 if thresholded.max() > 0 else 0  # Anomaly if max value > 0, otherwise normal
    #                 predicted_labels.append(predicted)

    #     # Calculate metrics
    #     accuracy = accuracy_score(true_labels, predicted_labels)
    #     precision = precision_score(true_labels, predicted_labels, zero_division=0)
    #     conf_matrix = confusion_matrix(true_labels, predicted_labels)
    #     class_report = classification_report(true_labels, predicted_labels)

    #     print(f"Accuracy: {accuracy:.4f}")
    #     print(f"Precision: {precision:.4f}")
    #     print("Confusion Matrix:")
    #     print(conf_matrix)
    #     print("Classification Report:")
    #     print(class_report)

    #     return accuracy, precision, conf_matrix, class_report

    def compare_images(model, test_loader, threshold, channels, thresholdType=None, FigSavePath=None, ModelName=None, display=True):
        """
        Compare the original and reconstructed images, and plot the reconstruction error.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        image_id = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device).float()
                labels = labels.cpu().numpy()

                # Forward pass
                outputs = model(images)
                # per-pixel error map, cropping off last 10 pixels
                reconstruction_error = ((images - outputs)**2).mean(dim=1)[:, :-10, :-10]

                pred = 0

                for i, lbl in enumerate(labels):
                    err_map = reconstruction_error[i].cpu().numpy()
                    # Thresholding
                    # if thresholdType in ("maxPix", "maxPixMSE"):
                    thresholded = np.where(err_map > threshold, err_map, 0)
                    # else:  # "MSE"
                    #     thresholded = np.where(err_map.mean() > threshold, err_map.mean(), 0)
                    pred = 1 if thresholded.max() > 0 else 0  # Anomaly if max value > 0, otherwise normal
                    # plotting
                    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
                    # for j in range(channels):
                    #     ax[j].imshow(images[i][j].cpu().numpy(), cmap='gray')
                    #     ax[j].set_title(f"Original Image {j+1}")
                    #     ax[j].axis('off')
                    ax[0].imshow(images[i][1].cpu().numpy(), cmap='gray')
                    ax[0].set_title(f"Original Image channel [1]")
                    ax[0].axis('off')

                    ax[1].imshow(outputs[i][1].cpu().numpy(), cmap='gray')
                    ax[1].set_title(f"Reconstruction Image channel [1]")
                    ax[1].axis('off')

                    ax[2].imshow(err_map, cmap='hot')
                    ax[2].set_title(f"Reconstruction error Image")
                    ax[2].axis('off')

                    ax[3].imshow(thresholded, cmap='hot')
                    ax[3].set_title(f"Reconstruction Error (Thresholded)")
                    ax[3].axis('off')

                    plt.suptitle(f"Label: {lbl}, Prediction: {pred}", fontsize=16)
                    if FigSavePath:
                        os.makedirs(os.path.dirname(FigSavePath), exist_ok=True)
                        plt.savefig(f"{FigSavePath}{ModelName}_label{lbl}Pred{pred}_comparison{image_id:04d}.png")
                    if display:
                        plt.show()
                    else:
                        plt.close()
                    image_id += 1

                
    
    def validate(model, val_loader, threshold, thresholdType=None, FigSavePath=None, ModelName=None, display=True):
        """
        Validate the model on the validation dataset and plot histogram, confusion matrix,
        classification report, and ROC curve.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        # storage
        true_labels = []
        predicted_labels = []
        anomaly_scores = []
        MSE = []
        MSEAnomaly = []

        with torch.no_grad():
            for images, labels in val_loader:  # val_loader yields (images, labels)
                images = images.to(device).float()
                labels = labels.cpu().numpy()

                # Forward pass
                outputs = model(images)
                # per-pixel error map, cropping off last 10 pixels
                reconstruction_error = ((images - outputs)**2).mean(dim=1)[:, :-10, :-10]

                # collect MSE lists for histogram
                for i, lbl in enumerate(labels):
                    err_map = reconstruction_error[i]
                    if lbl == 1:
                        if thresholdType in ("maxPix", "maxPixMSE"):
                            MSEAnomaly.append(err_map.max().item())
                        else:  # "MSE"
                            MSEAnomaly.append(err_map.mean().item())
                    else:
                        if thresholdType in ("maxPix", "maxPixMSE"):
                            MSE.append(err_map.max().item())
                        else:
                            MSE.append(err_map.mean().item())

                # predictions + scores
                err_np = reconstruction_error.cpu().numpy()
                for i, lbl in enumerate(labels):
                    if thresholdType in ("maxPix", "maxPixMSE"):
                        score = err_np[i].max()
                    else:  # "MSE"
                        score = err_np[i].mean()
                    anomaly_scores.append(score)
                    pred = 1 if score > threshold else 0
                    predicted_labels.append(pred)
                    true_labels.append(int(lbl))

        # 1) Histogram of scores
        plt.figure(dpi=250, figsize=(12, 8))
        plt.hist(MSE, bins=25, alpha=0.5, label='good')
        plt.hist(MSEAnomaly, bins=25, alpha=0.5, label='anomaly')
        plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=2, label='threshold')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title(f'Anomaly Score Histogram ({thresholdType})')
        plt.legend()
        plt.grid(True)
        if FigSavePath:
            os.makedirs(os.path.dirname(FigSavePath), exist_ok=True)
            plt.savefig(f"{FigSavePath}{ModelName}_{thresholdType}_hist.png")
        if display: plt.show()
        else: plt.close()

        # 2) Metrics and confusion matrix
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        class_report = classification_report(true_labels, predicted_labels)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)

        # plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(true_labels),
                    yticklabels=np.unique(true_labels))
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        if FigSavePath:
            plt.savefig(f"{FigSavePath}{ModelName}{thresholdType}_confusion.png")
        if display: plt.show()
        else: plt.close()

        # plot classification report as table
        rep_dict = classification_report(true_labels, predicted_labels, output_dict=True)
        rep_df = pd.DataFrame(rep_dict).transpose()
        plt.figure(figsize=(12, 6))
        plt.axis('off')
        plt.title('Classification Report', fontsize=16)
        tbl = plt.table(cellText=rep_df.values,
                        colLabels=rep_df.columns,
                        rowLabels=rep_df.index,
                        cellLoc='center',
                        loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.auto_set_column_width(col=list(range(len(rep_df.columns))))
        if FigSavePath:
            plt.savefig(f"{FigSavePath}{ModelName}{thresholdType}_class_report.png")
        if display: plt.show()
        else: plt.close()

        # 3) ROC curve
        fpr, tpr, ROCThresholds = roc_curve(true_labels, anomaly_scores)
        roc_auc = auc(fpr, tpr)

        distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
        opt_idx = np.argmin(distances)
        best_thresh_dist = ROCThresholds[opt_idx]
        print(f"Min‐distance threshold: {best_thresh_dist:.4f} (TPR={tpr[opt_idx]:.3f}, FPR={fpr[opt_idx]:.3f})")   

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], '--', label='chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({thresholdType})')
        plt.legend(loc='lower right')
        plt.grid(True)
        if FigSavePath:
            plt.savefig(f"{FigSavePath}{ModelName}_roc.png")
        if display: plt.show()
        else: plt.close()

        return accuracy, precision, conf_matrix, class_report, ROCThresholds, roc_auc

    # def validate(model, val_loader, threshold, thresholdType = None, FigSavePath=None, ModelName=None, display=True):
    #     """
    #     Validate the model on the validation dataset.
    #     """
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model = model.to(device)
    #     model.eval()

    #     criterion = nn.MSELoss()

    #     true_labels = []
    #     predicted_labels = []

    #     MSE = []
    #     MSEAnomaly = []

    #     with torch.no_grad():
    #         for images, labels in val_loader:  # Assuming val_loader returns (images, labels)
    #             images = images.to(device).float()
    #             labels = labels.cpu().numpy()  # Convert labels to numpy for metrics
    #             true_labels.extend(labels)

    #             # Forward pass
    #             outputs = model(images)
    #             reconstruction_error = ((images - outputs)**2).mean(axis=(1))[:,0:-10,0:-10]#((images - outputs) ** 2).mean(axis=(1))  # Per-image anomaly score
    #             #MSE.append(((images-outputs)**2).mean(axis=(1))[:,0:-10,0:-10].mean(axis=(1,2)))
    #             if (labels==1):
    #                 if thresholdType == "maxPix" or thresholdType == "maxPixMSE":
    #                     MSEAnomaly.append(torch.max(reconstruction_error).cpu().numpy())
    #                 elif thresholdType == "MSE":
    #                     MSEAnomaly.append((((images - outputs)**2).mean(axis=(1))[:,0:-10,0:-10].mean()).cpu().numpy())
    #                 else:
    #                     print("Error: thresholdType not defined")
    #             else:
    #                 if thresholdType == "maxPix" or thresholdType == "maxPixMSE":
    #                     MSE.append(torch.max(reconstruction_error).cpu().numpy())
    #                 elif thresholdType == "MSE":
    #                     MSE.append((((images - outputs)**2).mean(axis=(1))[:,0:-10,0:-10].mean()).cpu().numpy())
    #                 else:
    #                     print("Error: thresholdType not defined")
                    

    #             # Predict based on threshold
    #             # for i in range(len(reconstruction_error)):
    #             #     thresholded = np.where(reconstruction_error[i][0:-10,0:-10].cpu().numpy() > threshold, reconstruction_error[i][0:-10,0:-10].cpu().numpy(), 0)
    #             #     predicted = 1 if thresholded.max() > 0 else 0  # Anomaly if max value > 0, otherwise normal
    #             #     predicted_labels.append(predicted)
    #             if thresholdType == "maxPix" or thresholdType == "maxPixMSE":
    #                 thresholded = np.where(reconstruction_error.cpu().numpy() > threshold, reconstruction_error.cpu().numpy(), 0)
    #             elif thresholdType == "MSE":
    #                 thresholded = np.where(reconstruction_error.mean().cpu().numpy() > threshold, reconstruction_error.mean().cpu().numpy(), 0)
    #             else:
    #                 print("Error: thresholdType not defined")
    #             predicted = 1 if thresholded > 0 else 0  # Anomaly if max value > 0, otherwise normal
    #             predicted_labels.append(predicted)

    #     #Plot MSE Histogram with Threshold:
    #     plt.figure(dpi=250, figsize=(12, 8))
    #     plt.hist(MSE, bins=25, alpha=0.5, label='good')
    #     plt.hist(MSEAnomaly, bins=25, alpha=0.5, label='Anomaly')
    #     plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=2, label='Threshold')
    #     plt.xlabel('Anomaly Score')
    #     plt.ylabel('Frequency')
    #     plt.title(f'Anomaly score Histogram with {thresholdType} Threshold')
    #     plt.legend()
    #     plt.grid()
    #     if FigSavePath is not None:
    #         os.makedirs(os.path.dirname(FigSavePath), exist_ok=True)
    #         plt.savefig(f"{FigSavePath}{ModelName}{thresholdType}.png")
    #     if display:
    #         plt.show()
    #     else:
    #         plt.close()

    #     # Calculate metrics
    #     accuracy = accuracy_score(true_labels, predicted_labels)
    #     precision = precision_score(true_labels, predicted_labels, zero_division=0)
    #     conf_matrix = confusion_matrix(true_labels, predicted_labels)
    #     class_report = classification_report(true_labels, predicted_labels)

    #     print(f"Accuracy: {accuracy:.4f}")
    #     print(f"Precision: {precision:.4f}")
    #     print("Confusion Matrix:")
    #     print(conf_matrix)
    #     print("Classification Report:")
    #     print(class_report)

    #     # plot the confusion matrix
    #     plt.figure(figsize=(8,6))
    #     sns.heatmap(
    #         conf_matrix,
    #         annot=True,        # draw the numbers
    #         fmt='d',           # integer format
    #         cmap='Blues',      # same palette
    #         cbar=True,
    #         xticklabels=np.unique(true_labels),
    #         yticklabels=np.unique(true_labels)
    #     )
    #     plt.xlabel('Predicted label')
    #     plt.ylabel('True label')
    #     plt.title('Confusion Matrix')
    #     if FigSavePath is not None:
    #         os.makedirs(os.path.dirname(FigSavePath), exist_ok=True)
    #         plt.savefig(f"{FigSavePath}{ModelName}ConfusionMat.png")
    #     if display:
    #         plt.show()
    #     else:
    #         plt.close()

    #     # Example classification report as a string
    #     # Replace this with your actual `class_report`
    #     class_report_dict = classification_report(true_labels, predicted_labels, output_dict=True)

    #     # Convert the classification report to a DataFrame
    #     class_report_df = pd.DataFrame(class_report_dict).transpose()

    #     # Plot the table
    #     plt.figure(figsize=(12, 6))
    #     plt.axis("off")  # Turn off the axes
    #     plt.title("Classification Report", fontsize=16)
    #     table = plt.table(cellText=class_report_df.values,
    #                     colLabels=class_report_df.columns,
    #                     rowLabels=class_report_df.index,
    #                     cellLoc="center",
    #                     loc="center")

    #     table.auto_set_font_size(False)
    #     table.set_fontsize(10)
    #     table.auto_set_column_width(col=list(range(len(class_report_df.columns))))

    #     # Save the figure
    #     if FigSavePath is not None:
    #         os.makedirs(os.path.dirname(FigSavePath), exist_ok=True)
    #         plt.savefig(f"{FigSavePath}{ModelName}ClassReportTable.png")
    #     if display:
    #         plt.show()
    #     else:
    #         plt.close()

    #     # Make ROC curve

    #     return accuracy, precision, conf_matrix, class_report

    @staticmethod
    def get_threshold(train_loader, model):
        """
        Calculate the anomaly score threshold based on the training dataset.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        anomaly_scores = []

        with torch.no_grad():
            for images in train_loader:  # Iterate over batches
                if isinstance(images, (list, tuple)):
                    images = images[0]
                images = images.to(device).float()  # Move images to the device
                recon = model(images)  # Perform inference

                # Calculate reconstruction error for the batch
                recon_error = ((images - recon) ** 2).mean(dim=(1, 2, 3))  # Per-image anomaly score
                
                for i in range(len(recon_error)):
                    anomaly_scores.append(torch.max(recon_error[i]).cpu())  # Append the per-image anomaly score
                    print(f"Anomaly score for image {i}: {anomaly_scores[-1]}")

        # Calculate threshold: mean + 3 * std
        anomaly_scores = np.array(anomaly_scores)
        mean_score = np.mean(anomaly_scores)
        std_score = np.std(anomaly_scores)
        threshold = mean_score + 3 * std_score

        print(f"Threshold calculated: {threshold:.4f}, mean: {mean_score:.4f}, std: {std_score:.4f}")
        return threshold
    
    # TODO: Calculate the threshold from average of entire image score, not just the max score
    @staticmethod
    def get_threshold2(train_loader, model):
        """
        Calculate the anomaly score threshold based on the training dataset.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        RECON_ERROR=[]
        with torch.no_grad():
            for data in train_loader:
                data = data.cuda()
                recon = model(data)
                data_recon_squared_mean =  ((data-recon)**2).mean(axis=(1))[:,0:-10,0:-10].mean(axis=(1,2))
                
                RECON_ERROR.append(data_recon_squared_mean)
                
        RECON_ERROR = torch.cat(RECON_ERROR).cpu().numpy()

        mean_score = np.mean(RECON_ERROR)
        std_score = np.std(RECON_ERROR)
        threshold = mean_score + 3 * std_score

        # anomaly_scores = []

        # with torch.no_grad():
        #     for images in train_loader:  # Iterate over batches
        #         if isinstance(images, (list, tuple)):
        #             images = images[0]
        #         images = images.to(device).float()  # Move images to the device
        #         recon = model(images)  # Perform inference

        #         # Calculate reconstruction error for the batch
        #         recon_error = ((images - recon) ** 2).mean(dim=(1, 2, 3))  # Per-image anomaly score
                
        #         for i in range(len(recon_error)):
        #             anomaly_scores.append(torch.max(recon_error[i]).cpu())  # Append the per-image anomaly score
        #             print(f"Anomaly score for image {i}: {anomaly_scores[-1]}")

        # # Calculate threshold: mean + 3 * std
        # anomaly_scores = np.array(anomaly_scores)
        # mean_score = np.mean(anomaly_scores)
        # std_score = np.std(anomaly_scores)
        # threshold = mean_score + 3 * std_score

        print(f"Threshold calculated: {threshold:.4f}, mean: {mean_score:.4f}, std: {std_score:.4f}")
        return threshold
    
    @staticmethod
    def get_maxPixelThreshold(train_loader, model):
        """
        Calculate the anomaly score threshold based on the Max pixel intensity in the training dataset.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        RECON_ERROR=[]
        with torch.no_grad():
            for data in train_loader:
                data = data.cuda()
                recon = model(data)
                recon_error =  ((data-recon)**2).mean(axis=(1))[:,0:-10,0:-10]
                
                RECON_ERROR.append(np.max(recon_error.cpu().numpy()))

        threshold = np.max(RECON_ERROR)
        print(f"Threshold calculated: {threshold:.4f}")
        return threshold


    @staticmethod
    def get_maxPixelMSEThreshold(train_loader, model):
        """
        Calculate the anomaly score threshold based on the MSE of the max pixel intensities in the training dataset.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        RECON_ERROR=[]
        with torch.no_grad():
            for data in train_loader:
                data = data.cuda()
                recon = model(data)
                recon_error =  ((data-recon)**2).mean(axis=(1))[:,0:-10,0:-10]
                
                RECON_ERROR.append(np.max(recon_error.cpu().numpy()))

        mean_score = np.mean(RECON_ERROR)
        std_score = np.std(RECON_ERROR)
        threshold = mean_score + 3 * std_score
        print(f"Threshold calculated: {threshold:.4f}, mean: {mean_score:.4f}, std: {std_score:.4f}")
        return threshold

    @staticmethod
    def get_MSEThreshold(train_loader, model):
        """
        Calculate the anomaly score threshold based on the MSE of the training dataset.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        RECON_ERROR=[]
        with torch.no_grad():
            for data in train_loader:
                data = data.cuda()
                recon = model(data)
                data_recon_squared_mean =  ((data-recon)**2).mean(axis=(1))[:,0:-10,0:-10].mean(axis=(1,2))
                
                RECON_ERROR.append(data_recon_squared_mean)
                
        RECON_ERROR = torch.cat(RECON_ERROR).cpu().numpy()

        mean_score = np.mean(RECON_ERROR)
        std_score = np.std(RECON_ERROR)
        threshold = mean_score + 3 * std_score
        print(f"Threshold calculated: {threshold:.4f}, mean: {mean_score:.4f}, std: {std_score:.4f}")
        return threshold