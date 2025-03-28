import math
# import roboticstoolbox as rtb
from mpl_toolkits.mplot3d import Axes3D
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from scipy.spatial.transform import Rotation
import numpy as np
from spatialmath import SE3
import spatialmath as sm
import matplotlib.pyplot as plt
from photometric_positions import half_sphere_simple, position_to_pose, yaml_load_positions, robot_positions
from Camera.capture_image import capture_image, save_image
import pickle
import cv2
import os
#!/usr/bin/env python3

# ----------------------- Rotation matrix to Axis Angle ---------------------- #


def rotation_matrix_to_axis_angle(rotation_matrix):
    """
    Converts a 3x3 rotation matrix to an axis-angle rotation vector.
    Parameters:
        rotation_matrix (np.array): 3x3 rotation matrix
    Returns:
        axis angle (np.array): Axis-angle rotation vector (rx, ry, rz)
    """
    assert np.allclose(np.dot(rotation_matrix, rotation_matrix.T), np.eye(
        3)), "Input is not a valid rotation matrix"

    angle = np.arccos((np.trace(rotation_matrix) - 1) / 2)

    if angle == 0:
        return np.array([0, 0, 0])
    elif angle == np.pi:
        x = np.sqrt((rotation_matrix[0, 0] + 1) / 2)
        y = np.sqrt((rotation_matrix[1, 1] + 1) /
                    2) * np.sign(rotation_matrix[0, 1])
        z = np.sqrt((rotation_matrix[2, 2] + 1) /
                    2) * np.sign(rotation_matrix[0, 2])
        return np.array([x, y, z]) * angle
    else:
        rx = rotation_matrix[2, 1] - rotation_matrix[1, 2]
        ry = rotation_matrix[0, 2] - rotation_matrix[2, 0]
        rz = rotation_matrix[1, 0] - rotation_matrix[0, 1]
        axis = np.array([rx, ry, rz]) / (2 * np.sin(angle))
        return axis * angle


def collect_sample(ur_control, robot_positions, path, velocity=1.05, acceleration=1.4):
    """
    Move to the specified positions and collect data for each light source position.

    Args:
        ur_control (RTDEControlInterface): Interface to control the robot
        robot_positions (list): List of positions to move to
        path (str): Path to save the images
        velocity (float): Speed of the robot in rad/s
        acceleration (float): Acceleration of the robot in rad/s^2
    """
    n_channels = len(robot_positions.light_positions)
    if "channels" not in path:
        path = path + str(n_channels) + "_channels/"
        if not os.path.exists(path):
            os.makedirs(path)
    # Check if the connection is established
    if not ur_control.isConnected():
        print("Failed to connect to the robot")
        return False

    T_poses = []
    for pos in robot_positions.positions:
        T_poses.append(position_to_pose(pos, robot_positions.object_position))

    # Move to each position
    for i, T in enumerate(T_poses):
        # Print the current position
        print(f"Move to position {i+1}/{len(T_poses)}: \n{T.t}")
        AA_pose = list(T.t.tolist()) + \
            list(rotation_matrix_to_axis_angle(T.R))

        # Move to the position
        success = ur_control.moveJ_IK(
            AA_pose, speed=velocity, acceleration=acceleration)

        img_count = 0

        # Capture image
        if success and robot_positions.labels[i] == "Light":
            # Capture image
            img = capture_image(sim=True)
            if img is None:
                print("Failed to capture image")
                return False

            # Image filenmae
            count = len(os.listdir(path))
            light_number = count % n_channels
            sample_number = count // n_channels
            img_name = f"image_{sample_number}_light_{light_number}.png"

            # Save the image
            save_image(img, img_name, path)
            print(f"Image saved as {img_name}")

        if not success:
            print(f"Failed to move to position {i+1}")
            ur_control.disconnect()
            return False

    # Disconnect from the robot
    ur_control.disconnect()
    print("Movement completed successfully")
    return True


def trial_run(ur_control, robot_positions, velocity=0.2, acceleration=0.2):

    pose_T = []
    for pos in robot_positions.positions:
        pose_T.append(position_to_pose(pos, robot_positions.object_position))
    # Move to each position
    for i, T in enumerate(pose_T):
        print(f"[NEW] Next position: {i+1}/{len(pose_T)}: \n{T.t}")
        pose_AA = list(T.t.tolist()) + list(rotation_matrix_to_axis_angle(T.R))

        if robot_positions.labels[i] == "Light":
            input("[INFO] Press enter to move to the next light source")
        else:
            input("[INFO] Press enter to move to the next intermediate position")

        success = ur_control.moveJ_IK(
            pose_AA, speed=velocity, acceleration=acceleration)

        if not success:
            print(f"[ERROR] Failed to move to position {i+1}")
            ur_control.disconnect()
            return False

    return True


def main():
    # PARAMETERS
    rob_pos_filename = "robot_position.pkl"
    dataset_path = "irl_dataset/"

    # ---- #
    # Load the positions
    file = open(rob_pos_filename, "rb")
    robot_positions = pickle.load(file)
    file.close()
    print("[INFO] Positions loaded successfully")

    # Connect to the robot
    # ip = "192.168.1.30"
    ip = "localhost"
    ur_control = RTDEControlInterface(ip)
    ur_receive = RTDEReceiveInterface(ip)

    # Check if the connection is established
    if not ur_control.isConnected():
        print("[ERROR] Failed to connect to the robot")
        return False
    else:
        print("[INFO] Connected to the robot")

    # Set the speed and acceleration
    print("\n[PROMPT] Execute trial run (speed 0.2 rad/s)?")
    print("[PROMPT] If fast speed is desired enter here (rad/s):")
    speed = input(
        "[PROMPT] Otherwise press enter to continue or 'n' to skip trial run: ")

    if speed == "":
        velocity = 0.3
        acceleration = 0.2
    else:
        velocity = float(speed)
        acceleration = float(speed)

    print(f"\n[INFO] Speed: {velocity} rad/s")
    print(f"[INFO] Acceleration: {acceleration} rad/s^2\n")

    # Execute trial run
    if speed != "n":
        if trial_run(ur_control, robot_positions, velocity, acceleration):
            print("\n[INFO] Trial run completed successfully")
        else:
            print("\n[ERROR] Failed to complete trial run")
            ur_control.disconnect()
            return False

    # Begin data collection
    print("[PROMPT] Begin data collection?")
    n_its = input("[PROMPT] Enter number of repetitions: ")
    n_its = int(n_its)

    # Set the speed and acceleration
    velocity = 1
    acceleration = 1.4
    print(f"[INFO] Speed: {velocity} rad/s")
    print(f"[INFO]Acceleration: {acceleration} rad/s^2")

    # Collect data
    for i in range(n_its):
        print(f"\n[INFO] Iteration {i+1}/{n_its}")
        success = collect_sample(
            ur_control, robot_positions, dataset_path, velocity, acceleration)
        if not success:
            print("[INFO] Failed to complete data collection")
            ur_control.disconnect()
            return False


if __name__ == "__main__":
    main()
