# TODO: Implement preprocessing functions
# Resize
# Grayscale
# Layer images (8x256x256) jpg/ png
# Save image as one file
# Save as tensor
# Reconstructive surface normals
# Mosaic

import os
import cv2
import numpy as np


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

# Display images


def preview_images(images):
    for image in images:
        # Display image
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

# Convert images list to greyscale


def greyscale_images(images):
    greyscale_images = []
    for image in images:
        greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        greyscale_images.append(greyscale_image)
    return greyscale_images

# Layer images into groups of n images


def layer_images(images, n=4):
    # Assert that the number of images is divisible by n
    assert len(images) % n == 0, 'Number of images is not divisible by n'

    # Partition images into n groups
    partitioned_images = []
    for i in range(0, len(images), n):
        partitioned_images.append(images[i:i+n])
    partitioned_images = np.array(partitioned_images)

    print('Images imported succesfully with shape:')
    print(partitioned_images.shape)

    return partitioned_images


# Display layered images with image number and light position indicated
def preview_layered_images(layered_images):
    print('Layered images:', layered_images.shape)
    for layer_i, layer in enumerate(layered_images):
        for i, image in enumerate(layer):
            # Display image
            cv2.imshow(f'Image {layer_i}, Light pos {i}', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# Input path to the dataset
path = 'Datasets\Dataset001\Train'

images = load_images(path)
images = resize_images(images)
images = greyscale_images(images)
images = layer_images(images, 4)
tensor_images =


preview_layered_images(images)
