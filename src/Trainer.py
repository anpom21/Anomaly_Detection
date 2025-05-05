import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score

class Trainer:
    """
    Goal of this file: contain the training loop.
    """
    def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, save_path='best_model.pth', patience=5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model = model.to(device)

        criterion = nn.MSELoss()  # Reconstruction loss for autoencoders
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_loss = float('inf')
        epochs_without_improvement = 0

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

    def validate(model, val_loader, threshold):
        """
        Validate the model on the validation dataset.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for images, labels in val_loader:  # Assuming val_loader returns (images, labels)
                images = images.to(device).float()
                labels = labels.cpu().numpy()  # Convert labels to numpy for metrics
                true_labels.extend(labels)

                # Forward pass
                outputs = model(images)
                reconstruction_error = ((images - outputs) ** 2).mean(axis=(1))  # Per-image anomaly score

                # Predict based on threshold
                for i in range(len(reconstruction_error)):
                    thresholded = np.where(reconstruction_error[i][0:-10,0:-10].cpu().numpy() > 0.01, reconstruction_error[i][0:-10,0:-10].cpu().numpy(), 0)
                    predicted = 1 if thresholded.max() > 0 else 0  # Anomaly if max value > 0, otherwise normal
                    predicted_labels.append(predicted)

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        class_report = classification_report(true_labels, predicted_labels)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)

        return accuracy, precision, conf_matrix, class_report

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