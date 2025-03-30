import os
import PySpin
import matplotlib.pyplot as plt
import sys
import time
import numpy as np

def camera_init():
    """
    Initialize the camera:
      - Get system instance
      - Select the first available camera
      - Initialize the camera
      - Set acquisition mode to Continuous
      - Start image acquisition
    :return: Camera object, or None if no camera is found
    """
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()
    
    if num_cameras == 0:
        print('No cameras detected!')
        cam_list.Clear()
        system.ReleaseInstance()
        return None

    cam = cam_list[0]
    
    try:
        cam.Init()
        nodemap = cam.GetNodeMap()
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode (enumeration retrieval failed), exiting...')
            return None

        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsReadable(node_acquisition_mode_continuous):
            print('Unable to set acquisition mode (entry retrieval failed), exiting...')
            return None

        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
        print('Acquisition mode set to Continuous...')

        cam.BeginAcquisition()
        print('Camera acquisition started...')

        return cam

    except PySpin.SpinnakerException as ex:
        print('Camera initialization error: %s' % ex)
        return None


def take_image(cam):
    """
    Capture a single frame from the camera and return it as a numpy array.
    :param cam: Initialized and actively acquiring camera object
    :return: Image as a numpy array; None if acquisition fails
    """
    try:
        image_result = cam.GetNextImage(1000)  # Timeout set to 1000 milliseconds

        if image_result.IsIncomplete():
            print('Image acquisition incomplete, status: %d' % image_result.GetImageStatus())
            image_result.Release()
            return None
        else:
            image_data = image_result.GetNDArray()
            image_result.Release()
            return image_data

    except PySpin.SpinnakerException as ex:
        print('Error while capturing image: %s' % ex)
        return None


def camera_close(cam):
    """
    Close the camera and release resources.
    :param cam: Camera object
    """
    try:
        cam.EndAcquisition()
        cam.DeInit()
    except PySpin.SpinnakerException as ex:
        print('Error while closing camera: %s' % ex)
    
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam_list.Clear()
    system.ReleaseInstance()


if __name__ == '__main__':
    # Initialize the camera
    cam = camera_init()
    if cam is None:
        sys.exit(1)
    
    # Enable matplotlib interactive mode, create image window
    plt.ion()
    fig, ax = plt.subplots()
    print("Entering continuous acquisition mode. Close the image window or press Ctrl+C to exit the program...")
    
    try:
        # Continuously capture and display images
        while True:
            image = take_image(cam)
            if image is not None:
                ax.imshow(image, cmap='gray')
                plt.draw()
                plt.pause(0.001)  # Pause briefly to update display
                ax.clear()        # Clear current image for the next frame
            else:
                print("No image data captured")
            
            # Exit loop if the image window is closed
            if not plt.fignum_exists(fig.number):
                break

    except KeyboardInterrupt:
        print("Exit request detected, stopping acquisition...")

    finally:
        plt.ioff()
        plt.close(fig)
        camera_close(cam)
        print("Resources released, program exiting.")
