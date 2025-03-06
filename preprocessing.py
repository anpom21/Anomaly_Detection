# TODO: Implement preprocessing functions
# Resize
# Grayscale
# Layer images (8x256x256) jpg/ png
# Save image as one file
# Save as tensor
# Reconstructive surface normals
# Mosaic

# ---------------------------------------------------------------------------- #
#                                  How to use                                  #
# ---------------------------------------------------------------------------- #
"""
from preprocessing import preprocess_images, preview_layered_images
import numpy as np

PATH = 'path\to\images\Train' # ie 'Datasets\Dataset001\Train'

images = preprocess_images(PATH)
preview_layered_images(images)

# Load .npy file
images = np.load('layered_images.npy')
preview_layered_images(images)

"""
# ---------------------------------------------------------------------------- #
#                                    Import                                    #
# ---------------------------------------------------------------------------- #
import os
import cv2
import numpy as np
import torch
from torchvision import transforms

# ---------------------------------------------------------------------------- #
#                                   Function                                   #
# ---------------------------------------------------------------------------- #
# -------------------------------- Load images ------------------------------- #
# Load images from the dataset


def load_images(path):
    image_paths = []
    listdir = os.listdir(path)

    # Check if the directory is empty
    assert len(listdir) > 0, 'No images found in the directory'
    for image in listdir:
        image_paths.append(os.path.join(path, image))
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        images.append(image)
    return images

# ---------------------------------- Resize ---------------------------------- #
# Reisze images to 256x256


def resize_images(images, width=256, height=256):
    resized_images = []
    for image in images:
        if image.shape[0] != width or image.shape[1] != height:
            resized_image = cv2.resize(image, (width, height))
            print('Resized image shape:', resized_image.shape)
            resized_images.append(resized_image)
        else:
            resized_images.append(image)
    return resized_images

# --------------------------------- Greyscale -------------------------------- #
# Convert images list to greyscale


def greyscale_images(images):
    greyscale_images = []
    for image in images:
        greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        greyscale_images.append(greyscale_image)
    return greyscale_images

# ------------------------------- Layer images ------------------------------- #
# Layer images into groups of n images


def layer_images(images, n=4):
    # Assert that the number of images is divisible by n
    assert len(images) % n == 0, 'Number of images is not divisible by n'

    # Partition images into n groups
    partitioned_images = []
    for i in range(0, len(images), n):
        partitioned_images.append(images[i:i+n])
    partitioned_images = np.array(partitioned_images)

    return partitioned_images

# -------------------------- Preview Layered Images -------------------------- #
# Display layered images with image number and light position indicated


def preview_layered_images(layered_images, amount=4):
    assert type(layered_images) == np.ndarray, 'Input must be a numpy array'
    print(f'Displaying first {amount} layers.')
    for layer_i, layer in enumerate(layered_images[:amount]):
        for i, image in enumerate(layer):
            # Display image
            cv2.imshow(f'Image {layer_i}, Light {i}', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# ----------------------------- Preprocess images ---------------------------- #


def preprocess_images(folder_path, n=4, width=256, height=256):

    try:
        images = load_images(folder_path)
        images = resize_images(images, width, height)
        images = greyscale_images(images)
        images = layer_images(images, n)
        np.save('layered_images.npy', images)
        print('Images saved succesfully:', images.shape)
        return images
    except Exception as e:
        print('Error while preprocessing images:', e)
# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #


if __name__ == '__main__':

    # Test preprocessing functions
    # ------------------------- Input path to the dataset ------------------------ #
    path = 'Datasets\Dataset001\Train'

    # ------------------------ Run preprocessing functions ----------------------- #
    images = preprocess_images(path)

    # --------------------------- Save and load images --------------------------- #
    np.save('layered_images.npy', images[0:4])
    print('Images saved succesfully', images[0:4].shape)

    image_loaded = np.load('layered_images.npy')
    print('Loaded image shape:', image_loaded.shape)

    # Preview
    print('Images saved succesfully')

    preview_layered_images(image_loaded)
