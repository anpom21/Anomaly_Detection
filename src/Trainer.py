import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


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