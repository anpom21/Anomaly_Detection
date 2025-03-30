import os
import urllib.request
import tarfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F


def load_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    if not os.path.exists('data/carpet'):
        print("Downloading carpet dataset...")
        urllib.request.urlretrieve(
            "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz",
            "carpet.tar.xz"
        )
        with tarfile.open('carpet.tar.xz') as f:
            f.extractall('.')

    train_image_path = 'data/carpet/train'
    good_dataset = ImageFolder(root=train_image_path, transform=transform)

    x, y = good_dataset[0]
    print("Sample Image Shape: ", x.shape)
    print("Sample Label: ", y)

    total_samples = len(good_dataset)
    train_size = int(total_samples * 0.8)
    test_size = total_samples - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(good_dataset, [train_size, test_size])

    print("Total samples: ", total_samples)
    print("Training samples: ", len(train_dataset))
    print("Testing samples: ", len(test_dataset))

    BS = 16
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

    # 可视化部分批次图像
    image_batch, label_batch = next(iter(train_loader))
    print(f'Input batch shape: {image_batch.shape}')
    print(f'Label batch shape: {label_batch.shape}')
    grid = torchvision.utils.make_grid(image_batch[0:4], padding=5, nrow=4)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title('Good Samples')
    plt.axis('off')
    plt.show()

    return train_loader, test_loader, good_dataset.classes


class SimpleNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNet, self).__init__()
        # 卷积层部分
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 224, 224]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 32, 112, 112]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 112, 112]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 64, 56, 56]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [B, 128, 56, 56]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # 此层作为 Grad-CAM 目标层
            nn.MaxPool2d(2)  # [B, 128, 28, 28]
        )
        # 自适应池化层将特征图尺寸调整为 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平为 (batch_size, 128)
        x = self.fc(x)
        return x


def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=50):
    Validation_Loss = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            val_loss_sum = 0.0
            num_batches = 0
            for img, labels in test_loader:
                img = img.to(device)
                labels = labels.to(device)
                output = model(img)
                val_loss = criterion(output, labels)
                val_loss_sum += val_loss.item()
                num_batches += 1
            val_loss_avg = val_loss_sum / num_batches
            Validation_Loss.append(val_loss_avg)
        if epoch % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item(),
                                                                                val_loss_avg))
        # epoch_loss = running_loss / len(train_loader.dataset)


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


def print_test_predictions(model, device, test_loader, classes):
    model.eval()
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())
    print("\nTest Set Predictions:")
    for idx, (pred, gt) in enumerate(zip(predictions, ground_truth)):
        print(f"Sample {idx}: Prediction = {classes[pred]}, Ground Truth = {classes[gt]}")


def plot_feature_maps(model, device, test_loader):
    # 利用 hook 捕捉 conv_layers 中选定层的输出
    feature_maps = []

    def hook_fn(module, input, output):
        feature_maps.append(output)

    hook_handle = model.conv_layers[6].register_forward_hook(hook_fn)  # 选取第三个卷积块的 ReLU层
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images.to(device)
        _ = model(images)
    hook_handle.remove()

    feature = feature_maps[0]  # shape: (B, channels, H, W)
    num_channels = feature.shape[1]
    indices = torch.randperm(num_channels)[:10]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, idx in enumerate(indices):
        row = i // 5
        col = i % 5
        axes[row, col].imshow(feature[0, idx].detach().cpu().numpy(), cmap='gray')
        axes[row, col].set_title(f'Feature Map {idx.item()}')
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()


def compute_auc_roc(model, device, test_loader):
    model.eval()
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            # 假设二分类任务，取类别1的概率作为异常分数
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_scores.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    try:
        auc = roc_auc_score(all_labels, all_scores)
        print(f"AUC-ROC: {auc:.4f}")
    except Exception as e:
        print("Error computing AUC-ROC:", e)


def generate_heatmap(model, device, image, target_class=None):
    model.eval()
    activation = None
    gradients = None

    # 前向 hook 用于捕捉特征图
    def forward_hook(module, input, output):
        nonlocal activation
        activation = output.detach()

    # 反向 hook 用于捕捉梯度
    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    # 选择 conv_layers 中最后一个 ReLU 层（本例中为索引 6）作为目标层
    target_layer = model.conv_layers[6]
    f_handle = target_layer.register_forward_hook(forward_hook)
    b_handle = target_layer.register_backward_hook(backward_hook)

    image = image.unsqueeze(0).to(device)
    output = model(image)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    # 对目标类别得分进行反向传播
    model.zero_grad()
    loss = output[0, target_class]
    loss.backward()

    # 移除 hook
    f_handle.remove()
    b_handle.remove()

    # 计算每个通道的权重：对梯度做全局平均池化
    weights = gradients.mean(dim=(2, 3))
    # 加权求和得到热力图
    weighted_activation = (weights.unsqueeze(2).unsqueeze(3) * activation)
    heatmap = weighted_activation.sum(dim=1).squeeze(0)
    heatmap = F.relu(heatmap)
    heatmap -= heatmap.min()
    if heatmap.max() != 0:
        heatmap /= heatmap.max()
    # 上采样到原图尺寸（224x224）
    heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    heatmap = heatmap.squeeze().cpu().numpy()
    return heatmap, target_class


def plot_prediction_heatmap(model, device, test_loader):
    # 从测试集中选取一个样本
    images, labels = next(iter(test_loader))
    image = images[0]
    label = labels[0].item()
    heatmap, predicted_class = generate_heatmap(model, device, image)

    # 将归一化后的图像反归一化（便于可视化）
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # unnorm = transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)],
    #                              std=[1/s for s in std])
    # image_unnorm = unnorm(image).clamp(0,1).permute(1,2,0).cpu().numpy()

    # 显示原图与热力图叠加
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image.clamp(0, 1).permute(1, 2, 0).cpu().numpy())
    plt.title(f"Original Image (Label: {image})")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image.clamp(0, 1).permute(1, 2, 0).cpu().numpy())
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title(f"Grad-CAM Heatmap (Pred: {predicted_class})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, classes = load_data()
    num_classes = len(classes)

    model = SimpleNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    print("Start training SimpleNet...")
    train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=num_epochs)

