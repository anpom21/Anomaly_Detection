import matplotlib.pyplot as plt
import numpy as np
import torch
import backbones
from simplenet import SimpleNet
from mvtec import MVTecDataset, DatasetSplit

if __name__ == '__main__':
    TRAIN = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone_name = 'wideresnet50'
    layers_to_extract_from = ['layer2', 'layer3']
    input_shape = (3, 224, 224)
    pretrain_embed_dimension = 1536
    target_embed_dimension = 1536
    patchsize = 3
    embedding_size = 3
    meta_epochs = 5
    aed_meta_epochs = 1
    gan_epochs = 4
    noise_std = 0.015
    dsc_layers = 4
    dsc_hidden = 1024
    dsc_margin = 0.5
    dsc_lr = 0.0002
    auto_noise = 0
    train_backbone = False
    cos_lr = True
    pre_proj = 1
    proj_layer_type = 0
    mix_noise = 1

    backbone = backbones.load(backbone_name)

    model = SimpleNet(device)
    model.load(backbone=backbone, layers_to_extract_from=layers_to_extract_from, device=device, input_shape=input_shape,
               pretrain_embed_dimension=pretrain_embed_dimension, target_embed_dimension=target_embed_dimension,
               patchsize=patchsize, embedding_size=embedding_size, meta_epochs=meta_epochs,
               aed_meta_epochs=aed_meta_epochs, gan_epochs=gan_epochs, noise_std=noise_std, dsc_layers=dsc_layers,
               dsc_hidden=dsc_hidden, dsc_margin=dsc_margin, dsc_lr=dsc_lr, auto_noise=auto_noise,
               train_backbone=train_backbone, cos_lr=cos_lr, pre_proj=pre_proj, proj_layer_type=proj_layer_type,
               mix_noise=mix_noise)
    model.set_model_dir('model', 'simplenet')

    data_path = 'data'
    classname = 'Dataset001'
    resize = 224
    train_val_split = 1
    imagesize = 224
    seed = 1
    rotate_degrees = 0
    translate = 0.0
    brightness_factor = 0.0
    contrast_factor = 0.0
    saturation_factor = 0.0
    gray_p = 0.0
    h_flip_p = 0.0
    v_flip_p = 0.0
    scale = 0.0
    augment = True
    train_dataset = MVTecDataset(source=data_path, classname=classname, resize=resize, train_val_split=train_val_split,
                                 imagesize=imagesize, split=DatasetSplit.TRAIN, seed=seed,
                                 rotate_degrees=rotate_degrees, translate=translate,
                                 brightness_factor=brightness_factor, contrast_factor=contrast_factor,
                                 saturation_factor=saturation_factor, gray_p=gray_p, h_flip_p=h_flip_p,
                                 v_flip_p=v_flip_p, scale=scale, augment=augment)
    test_dataset = MVTecDataset(source=data_path, classname=classname, resize=resize, imagesize=imagesize,
                                split=DatasetSplit.TEST, seed=seed)

    batch_size = 8
    num_workers = 2
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, prefetch_factor=2, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers, prefetch_factor=2, pin_memory=True)

    # i_auroc, p_auroc, pro_auroc = model.train(train_dataloader, test_dataloader)
    # print('i_auroc: ', i_auroc)
    # print('p_auroc: ', p_auroc)
    # print('pro_auroc: ', pro_auroc)
    if TRAIN:
        exit()

    train_scores = model.test_train(train_dataloader)

    # for ok images
    best_threshold = np.mean(train_scores) + 2 * np.std(train_scores)
    plt.figure(1)
    plt.hist(train_scores, bins=50)
    plt.vlines(x=best_threshold, ymin=0, ymax=30, colors='r')

    test_scores, test_segmentations, labels_gt = model.test(test_dataloader, True)
    plt.figure(2)
    plt.hist(test_scores, bins=50)
    plt.vlines(x=best_threshold, ymin=0, ymax=30, colors='r')

    segm_map = torch.nn.functional.interpolate(
        torch.from_numpy(np.array(test_segmentations[0])).view(1, 1, 224, 224),
        size=(224, 224),
        mode='bilinear'
    )
    plt.figure(3)
    plt.imshow(segm_map.squeeze(), cmap='jet')

    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score

    #
    y_score, y_segmentations, y_true = model.test(test_dataloader, False)

    # Calculate AUC-ROC score
    auc_roc_score = roc_auc_score(y_true, y_score)
    print("AUC-ROC Score:", auc_roc_score)

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    plt.figure(4)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_roc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    f1_scores = [f1_score(y_true, y_score >= threshold) for threshold in thresholds]

    # Select the best threshold based on F1 score
    best_threshold = thresholds[np.argmax(f1_scores)]

    print(f'best_threshold = {best_threshold}')

    # Generate confusion matrix
    cm = confusion_matrix(y_true, (y_score >= best_threshold).astype(int))
    # plt.figure(5)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['OK', 'NOK'])
    disp.plot()

    # Printout the prediction on the testset
    from PIL import Image
    from torchvision.transforms import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    class_label = ['OK', 'NOK']

    for i in range(len(test_dataset)):
        test_image = transform(Image.open(test_dataset[i]['image_path']))

        score = y_score[i]
        fault_type = test_dataset[i]['anomaly']
        y_pred_image = 1 * (y_score[i] >= best_threshold)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(test_image.permute(1, 2, 0))
        plt.title(f'fault type: {fault_type}')

        plt.subplot(1, 3, 2)
        plt.imshow(y_segmentations[i], cmap='jet', vmin=best_threshold, vmax=best_threshold * 2)
        plt.title(f'Anomaly score: {y_score[i] / best_threshold:0.4f} || {class_label[y_true[i]]}')

        plt.subplot(1, 3, 3)
        plt.imshow(y_segmentations[i] > best_threshold * 1.25, cmap='gray')
        plt.title(f'segmentation map')

        plt.show()

    plt.show()
