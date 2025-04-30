import PySpin
import numpy as np
import cv2


def camera_init():
    """
    Initializes the camera and returns the camera object.

    :return: Initialized camera object or None if no camera is found.
    """
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        print("No cameras detected!")
        cam_list.Clear()
        system.ReleaseInstance()
        return None
    cam = cam_list.GetByIndex(0)  # Assuming only one camera is connected
    cam.Init()
    cam_list.Clear()
    return system


def take_image():
    """
    Captures an image from the initialized camera and returns it as a numpy array.

    :param cam: Initialized camera object
    :return: Image as numpy array or None if failed.
    """
    system = camera_init()
    cam_list = system.GetCameras()
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


def main():
    print("Initializing camera...")

    # print("Camera initialized:", camera)
    while True:
        print("Acquiring image...")
        image = take_image()

        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if image is not None:
            print("Image captured successfully, shape:", image.shape)

    print("Done.")


if __name__ == "__main__":
    main()
