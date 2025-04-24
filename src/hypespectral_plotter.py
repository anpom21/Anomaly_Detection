import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_hyperspectral_image(data, recon, recon_error, title="Hyperspectral Image"):
    plt.figure(dpi=250)
    fig, ax = plt.subplots(3, data[0].cpu().numpy().shape(0), figsize=(5*4, 4*4))
    for i in range(3):
        ax[0, i].imshow(data[i].cpu().numpy().transpose((1, 2, 0)))
        ax[1, i].imshow(recon[i].cpu().numpy().transpose((1, 2, 0)))
        ax[2, i].imshow(recon_error[i][0:-10,0:-10].cpu().numpy(), cmap='jet',vmax= torch.max(recon_error[i])) #[0:-10,0:-10]
        ax[0, i].axis('OFF')
        ax[1, i].axis('OFF')
        ax[2, i].axis('OFF')
    plt.show()