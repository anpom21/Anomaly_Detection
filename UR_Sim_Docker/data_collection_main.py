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
import pickle
import cv2
import os
import PySpin
from photometric_positions import half_sphere_oriented, position_to_pose, robot_positions
from Camera.capture_image import capture_image, save_image
from UR_control import trial_run, collect_sample
#!/usr/bin/env python3

# ----------------------- Rotation matrix to Axis Angle ---------------------- #


def main():
    # PARAMETERS
    rob_pos_filename = "robot_position.pkl"
    dataset_path = "irl_dataset/"

    # Connect to the robot
    ip_real = "192.168.1.30"
    ip_sim = "localhost"

    # ---- #
    # ---------------------------- Load the positions ---------------------------- #
    file = open(rob_pos_filename, "rb")
    robot_positions = pickle.load(file)
    file.close()
    print("[INFO] Positions loaded successfully")

    # Check if the pose is reachable
    robot = rtb.models.UR5()
    for pos in robot_positions.light_positions:
        T = position_to_pose(pos, robot_positions.object_position)

        sol = robot.ikine_LM(T, q0=robot_positions.home_position)
        if not sol.success:
            print(f"[ERROR] Pose {pos} is not reachable")
            return False

    print("[INFO] All poses are reachable")

    # --------------------------- Execute sim trial run -------------------------- #
    sim_trial = input("[PROMPT] Execute simulation trial run? (y/n)")

    if sim_trial.lower() != "n":
        sim_ur_control = RTDEControlInterface(ip_sim)
        sim_ur_receive = RTDEReceiveInterface(ip_sim)
        # Check connection
        if sim_ur_control.isConnected():
            print("[INFO] Connected to the simulation robot")
        else:
            print("[ERROR] Failed to connect to the simulation robot")
            return False
        velocity = 0.8
        acceleration = 0.2
        if trial_run(sim_ur_control, robot_positions, velocity, acceleration):
            print("\n[INFO] Simulation trial run completed successfully")
        else:
            print("\n[ERROR] Failed to complete simulation trial run")
            sim_ur_control.disconnect()
            return False
        sim_ur_control.disconnect()
        sim_ur_receive.disconnect()

    # # ---------------------- Set the speed and acceleration ---------------------- #

    trial_run_input = input(
        "\n[PROMPT] Execute trial run on real robot? (y/n)")
    # print("[PROMPT] If fast speed is desired enter here (rad/s):")
    # speed = input(
    #     "[PROMPT] Otherwise press enter to continue or 'n' to skip trial run: ")

    # if speed == "":
    #     velocity = 0.3
    #     acceleration = 0.2
    # else:
    #     velocity = float(speed)
    #     acceleration = float(speed)

    # print(f"\n[INFO] Speed: {velocity} rad/s")
    # print(f"[INFO] Acceleration: {acceleration} rad/s^2\n")
    # --------------------------- Connect to the robot --------------------------- #
    real_ur_control = RTDEControlInterface(ip_real)
    real_ur_receive = RTDEReceiveInterface(ip_real)
    # Check connection
    if real_ur_control.isConnected():
        print("[INFO] Connected to the real robot")
    else:
        print("[ERROR] Failed to connect to the real robot")
        return False
    # ----------------------------- Execute trial run ---------------------------- #
    if trial_run_input != "n":
        velocity = 0.1
        acceleration = 0.2
        trial_succes = trial_run(
            real_ur_control, robot_positions, velocity, acceleration)
        if trial_succes:
            print("\n[INFO] Trial run completed successfully")
        else:
            print("\n[ERROR] Failed to complete trial run")
            real_ur_control.disconnect()
            return False

    # --------------------------- Begin data collection -------------------------- #
    # Prompt the user to start data collection
    print("[PROMPT] Begin data collection?")
    n_its = input("[PROMPT] Enter number of repetitions: ")
    try:
        n_its = int(n_its)
    except ValueError:
        print("[ERROR] Invalid input, using default value of 1")
        n_its = 1

    # Set the speed and acceleration for data collection
    velocity = 0.6
    acceleration = 0.5
    print(f"[INFO] Speed: {velocity} rad/s")
    print(f"[INFO]Acceleration: {acceleration} rad/s^2")

    # Begin data collection
    camera_system = PySpin.System.GetInstance()
    print("[INFO] Beginning data collection")
    for i in range(n_its):
        print(f"\n[INFO] Iteration {i+1}/{n_its}")
        change_speed = input(
            "[PROMPT] Change speed and acceleration? (type yes to change)")
        if change_speed.lower() == "yes":
            velocity = input(
                "[PROMPT] Enter new speed (rad/s): ")
            try:
                velocity = float(velocity)
            except ValueError:
                print("[ERROR] Invalid input, using default value of 0.3 rad/s")
                velocity = 0.3
            acceleration = input(
                "[PROMPT] Enter new acceleration (rad/s^2): ")
            try:
                acceleration = float(acceleration)
            except ValueError:
                print("[ERROR] Invalid input, using default value of 0.5 rad/s^2")
                acceleration = 0.5
        success = collect_sample(
            real_ur_control, camera_system, robot_positions, dataset_path, velocity, acceleration)
        if not success:
            print("[INFO] Failed to complete data collection")
            real_ur_control.disconnect()
            return False

    real_ur_control.disconnect()
    camera_system.ReleaseInstance()
    print("[INFO] Data collection completed successfully")


if __name__ == "__main__":
    main()
