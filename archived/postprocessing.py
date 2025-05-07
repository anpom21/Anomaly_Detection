import cv2
import os

import matplotlib.pyplot as plt
import numpy as np



def ThresholdFilter(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # Display the image
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.show()
    whiteCnt = []
    for i in range(0, image.max(), 1):
        image_threshold = cv2.threshold(image, i, 255, cv2.THRESH_BINARY )[1]
        whiteCnt.append(cv2.countNonZero(image_threshold))
    plt.plot(range(0, image.max(), 1), whiteCnt)
    plt.title("Threshold vs White Pixel Count")
    plt.xlabel("Threshold Value")
    plt.ylabel("White Pixel Count")
    plt.ylim([0, 500])
    plt.show()

def ThresholdFilterImage(image):
    image = cv2.morphologyEx(image, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_OPEN, (3, 3)))
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # Display the image
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.show()
    for i in range(0, image.max(), 1):
        image_threshold = cv2.threshold(image, i, 255, cv2.THRESH_BINARY )[1]
        whiteCnt = cv2.countNonZero(image_threshold)
        plt.imshow(image_threshold, cmap='gray')
        plt.title("Threshold: " + str(i) + " White Pixel Count: " + str(whiteCnt))
        plt.axis('off')
        plt.pause(0.1)
    plt.close()


def confusionMatrix(image, threshold):

    image = cv2.GaussianBlur(image, (5, 5), 0)
    image_threshold = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY )[1]
    if cv2.countNonZero(image_threshold) > 50:
        return 1
    else:
        return 0
def Threshold():
    # Get every image in the folder "test/"
    image_folder = "test"
    
    truth = [1] * 48 + [0] * 50
    pred = []
    for i in range(0, 255, 1):
        conf = []
        print("Threshold: ", i)
        j = 0
        for filename in os.listdir(image_folder):
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                image_path = os.path.join(image_folder, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                result = confusionMatrix(image, i)
                if result == truth[j]:
                    conf.append(1)
                else:
                    conf.append(0)
                j += 1
        pred.append(conf)
    return pred

def TestFilter(image):
    # image = cv2.morphologyEx(image, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

    kernel_size = 5
    image = cv2.medianBlur(image, kernel_size)
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    image = blurred_image



    # Display the image
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.show()


    # Calculate the histogram
    print("Maximum pixel intensity in the image:", image.max())
    plt.hist(image.ravel(), bins=256, range=[0, image.max()], color='blue', alpha=0.7)
    plt.ylim([0, 500])
    plt.title("Histogram of Pixel Intensities")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


# Path to the image
image_path = os.path.join("test", "error_48.png")

# Load the image in grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# ThresholdFilterImage(image)

# ----------------------- Plot defect and normal images ---------------------- #
images = []
images.append(cv2.imread('test/error_1.png'))
images.append(cv2.imread('test/error_2.png'))
images.append(cv2.imread('test/error_48.png'))
images.append(cv2.imread('test/error_49.png'))

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, img, title in zip(axes, images, ['Defect 1', 'Defect 2', 'Normal 1', 'Normal 2']):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
# confusionMatrix()