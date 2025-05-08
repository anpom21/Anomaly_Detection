from inverseKinematicsUR5 import InverseKinematicsUR5, transformRobotParameter
import numpy as np
from math import pi
import spatialmath as sm
import roboticstoolbox as rtb


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


# ---------------------------- Generate goal pose ---------------------------- #
goal_q = [0.32826396128913693, -1.1278131621442604, 0.2913550630973498, -
          0.08764109578770451, 0.6135990115288061, 0.14006411878463432]
goal_pose = [-0.4600239814071158, -0.38352159843880684, 0.8232937409239127,
             1.0657471417228515, 0.3367109420998421, -0.6797240954649697]
v = goal_pose[3:] / np.linalg.norm(goal_pose[3:])
theta = np.linalg.norm(goal_pose[3:])
print("v:", v)
print("theta:", theta)
R = sm.base.angvec2r(theta, v, unit='rad')
# Conert R back into axis-angle
axis_angle = rotation_matrix_to_axis_angle(R)
print("Axis-angle:", axis_angle)
t = np.array(goal_pose[:3])
print("t:", t)
print("R:", R)
T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = t

# Convert to SE3 object
# T_tool = np.eye(4)
# T_tool[:3, 3] = [0, 0, 0.185]
#
# T = T @ np.linalg.inv(T_tool)
T = sm.SE3(T)
print("Goal T:", T)
print("Goal q:", goal_q)

print("FK of joint angles:", transformRobotParameter(goal_q))
# ------------------------------------- v ------------------------------------ #


theta0 = [0.6, 0.2, 0.5, 1.2, 0.1, -0.1]
theta = [0.7, 0.3, 0.2, 1.0, 0.5, 0.1]
gd = transformRobotParameter(theta)
print("T:", T)
T_arr = T._A
print("T_list:", T_arr)
# print(gd)
gd = T_arr
joint_weights = [1, 1, 1, 1, 1, 1]
ik = InverseKinematicsUR5()
ik.setJointWeights(joint_weights)
ik.setJointLimits(-pi, pi)
# ik.setEERotationOffsetROS()
# print(ik.solveIK(gd))
# print(ik.findClosestIK(gd, theta0))

robot = rtb.models.UR5()
# Print the solutions
for i, q in enumerate(ik.solveIK(gd)):
    print(f"Solution {i+1}: {q}")
    # Print the forward kinematics for the solution
    T_sol = transformRobotParameter(q)
    print(f"Forward Kinematics for Solution {i+1}: {T_sol}")

# for i, q in enumerate(ik.solveIK(gd)):
#     print(q)

#     print(f"\nSolution {i+1}: {q}")
#     # Print the forward kinematics for the solution
#     T_sol = robot.fkine(q)
#     print(f"Forward Kinematics for Solution {i+1}: {T_sol}")
#     robot.plot(q,  backend='pyplot', block=True)
#     # input("Press Enter to continue...")
# After solving for all solutions
all_solutions = ik.solveIK(T._A)

# Set your known joint config (from the simulator)
goal_q = np.array([0.45331127, -1.0165405, 0.85382952,
                  0.49329873, 0.90134015, 0.19516498])
closest_q = ik.findClosestIK(T._A, goal_q)

print("Goal q:", goal_q)
print("Closest IK solution:", closest_q)

errors = [np.linalg.norm(q - goal_q) for q in all_solutions]
for i, err in enumerate(errors):
    print(f"Solution {i+1} error: {err:.6f}")
