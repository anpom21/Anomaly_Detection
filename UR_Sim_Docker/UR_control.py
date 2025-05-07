import math
import roboticstoolbox as rtb
from mpl_toolkits.mplot3d import Axes3D
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from scipy.spatial.transform import Rotation
import numpy as np
from spatialmath import SE3
import spatialmath as sm
import matplotlib.pyplot as plt
from photometric_positions import half_sphere_oriented, position_to_pose, yaml_load_positions, robot_positions
from Camera.capture_image import capture_image, save_image
import pickle
import cv2
import os
import PySpin
#!/usr/bin/env python3

# ----------------------- Rotation matrix to Axis Angle ---------------------- #


def rotation_matrix_to_axis_angle(R):
    """
    Convert rotation matrix to axis-angle (rx, ry, rz).
    """
    assert np.allclose(np.dot(R, R.T), np.eye(
        3), atol=1e-6), "Invalid rotation matrix"

    angle = np.arccos((np.trace(R) - 1) / 2)

    if np.isclose(angle, 0):
        return np.array([0.0, 0.0, 0.0])
    elif np.isclose(angle, np.pi):
        eigvals, eigvecs = np.linalg.eig(R)
        axis = eigvecs[:, np.isclose(eigvals, 1.0)].flatten().real
        axis = axis / np.linalg.norm(axis)
        return axis * angle
    else:
        rx = R[2, 1] - R[1, 2]
        ry = R[0, 2] - R[2, 0]
        rz = R[1, 0] - R[0, 1]
        axis = np.array([rx, ry, rz]) / (2 * np.sin(angle))
        return axis * angle


def collect_sample(ur_control, camera_system, robot_positions, path, velocity=1.05, acceleration=1.4):
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
    for i, pos in enumerate(robot_positions.positions):
        if robot_positions.labels[i] == "Light":
            T_poses.append(position_to_pose(
                pos, robot_positions.object_position))
        else:
            # Intermediates are joint angles
            T_poses.append(pos)

    # Move to each position
    for i, T in enumerate(T_poses):
        pose = 0
        # Print the current position
        if robot_positions.labels[i] == "Light":
            print(
                f"[NEW] Next position: {i+1}/{len(T_poses)}: \n{T.t}\n {T.R}")
            pose = list(T.t.tolist()) + \
                list(rotation_matrix_to_axis_angle(T.R))
            print(f"[INFO] Axis angle: {pose}")
        else:
            # Intermediates are joint angles
            print("[INFO] INTERMEDIATE")
            pose = T

        # Move to the position
        if robot_positions.labels[i] == "Light":
            input("[INFO] Press enter to move to the next light source")
            success = ur_control.moveJ_IK(
                pose, speed=velocity, acceleration=acceleration)
        else:
            input("[INFO] Press enter to move to the next intermediate position")
            print("[INFO] Joint angles:", pose)
            success = ur_control.moveJ(
                pose, speed=velocity, acceleration=acceleration)

        img_count = 0

        # Capture image
        if success and robot_positions.labels[i] == "Light":
            # Capture image

            img = capture_image(camera_system)
            if img is None:
                print("[ERROR] Failed to capture image")
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
    print("Movement completed successfully")
    return True


def trial_run(ur_control, robot_positions, velocity=0.2, acceleration=0.2):

    pose_T = []
    for i, pos in enumerate(robot_positions.positions):
        if robot_positions.labels[i] == "Light":
            pose_T.append(position_to_pose(
                pos, robot_positions.object_position))
        else:
            # Intermediates are joint angles
            pose_T.append(pos)
    # Move to each position
    for i, T in enumerate(pose_T):
        pose = 0
        if robot_positions.labels[i] == "Light":
            print(f"[NEW] Next position: {i+1}/{len(pose_T)}: \n{T.t}\n {T.R}")
            pose = list(T.t.tolist()) + \
                list(rotation_matrix_to_axis_angle(T.R))
            print(f"[INFO] Axis angle: {pose}")
        else:
            # Intermediates are joint angles
            print("[INFO] INTERMEDIATE")
            pose = T

        # Move to the position
        if robot_positions.labels[i] == "Light":
            input("[INFO] Press enter to move to the next light source")
            success = ur_control.moveJ_IK(
                pose, speed=velocity, acceleration=acceleration)
        else:
            input("[INFO] Press enter to move to the next intermediate position")
            success = ur_control.moveJ(
                pose, speed=velocity, acceleration=acceleration)

        if not success:
            print(f"[ERROR] Failed to move to position {i+1}")
            ur_control.disconnect()
            return False

    return True


def main():
    # Load the robot model
    robot = rtb.models.UR5()
    # Connect to the robot
    ur_control = RTDEControlInterface("localhost")
    ur_receive = RTDEReceiveInterface("localhost")

    current_q = ur_receive.getActualQ()
    print(f"Current joint angles: {current_q}")


if __name__ == "__main__":
    main()
