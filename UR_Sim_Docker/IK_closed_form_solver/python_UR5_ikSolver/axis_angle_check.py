import numpy as np
from math import pi
import spatialmath as sm


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


goal_pose = [-0.5008937581618724, -0.46103523661830254, 0.4677549449120001,
             1.6745510549036335, -0.7335665903030507, -0.01112144345787534]

org_axis_angle = goal_pose[3:]
print("Original axis-angle:", org_axis_angle)
v = goal_pose[3:] / np.linalg.norm(goal_pose[3:])
theta = np.linalg.norm(goal_pose[3:])
print("v:", v)
print("theta:", theta)
R = sm.base.angvec2r(theta, v, unit='rad')
print("R:", R)
# Conert R back into axis-angle
axis_angle = sm.base.tr2angvec(R)
print("Axis-angle:", axis_angle)
# Combined axis-angle
axis_angle_combined = axis_angle[0]*axis_angle[1]
print("Combined axis-angle:", axis_angle_combined)
# Check if the axis-angle is equal to the original axis-angle
