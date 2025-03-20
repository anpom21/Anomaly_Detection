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
from kinematic_functions import position_to_pose, pose_vector_to_se3, rotation_matrix_to_axis_angle, student_IK, move_to_start_configuration


robot_ip = "192.168.1.30"
rtde_c = RTDEControlInterface(robot_ip)
rtde_r = RTDEReceiveInterface(robot_ip)

print("Connected to robot")

print("Robot position: ", rtde_r.getActualTCPPose())
print("Robot joint positions: ", rtde_r.getActualQ())