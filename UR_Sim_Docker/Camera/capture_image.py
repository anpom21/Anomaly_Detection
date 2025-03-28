import PySpin
import numpy as np
import cv2
import os


# ------------------------------- Capture Image ------------------------------ #
def capture_image():
    """
    Captures an image from the initialized camera and returns it as a numpy array.

    :param cam: Initialized camera object
    :return: Image as numpy array or None if failed.
    """
    # Initialize camera
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    # Check if camera is found
    if cam_list.GetSize() == 0:
        print("No cameras detected!")
        cam_list.Clear()
        system.ReleaseInstance()
        # Throw error
        raise Exception("No cameras detected!")
        return None
    cam = cam_list.GetByIndex(0)
    cam.Init()
    cam.BeginAcquisition()

    image_result = cam.GetNextImage(1000)

    if image_result.IsIncomplete():
        print(f"Image incomplete with status {image_result.GetImageStatus()}.")
        cam.EndAcquisition()
        return None

    image_data = image_result.GetNDArray()
    image_result.Release()

    cam.EndAcquisition()
    return image_data


def save_image(image, filename, path):
    """
    Saves the image to the specified path with the specified filename.

    :param image: Image to save
    :param filename: Name of the file
    :param path: Path to save the image
    """

    # Check if the path exists
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the image
    cv2.imwrite(path + filename, image)


def main():
    print("Initializing camera...")

    # print("Camera initialized:", camera)
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        print("Acquiring image...")
        try:
            image = capture_image()
        except:

            print("Camera initialized")
            _, image = cap.read()

        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save image
        n_channels = 2
        path = "irl_dataset/"
        if "channels" not in path:
            n_channels = 2
            path = path + str(n_channels) + "_channels/"
            if not os.path.exists(path):
                os.makedirs(path)
        count = len(os.listdir(path))
        light_number = count % n_channels
        sample_number = count // n_channels
        img_name = f"image_{sample_number}_light_{light_number}.png"
        save_image(image, img_name, path)
        print("Imaged saved at: ", path)
        if image is not None:
            print("Image captured successfully, shape:", image.shape)


if __name__ == "__main__":
    main()
