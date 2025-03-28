
import numpy as np
# import roboticstoolbox as rtb
from spatialmath import SE3
from scipy.spatial.transform import Rotation
import spatialmath as sm
import time

# --------------------- UR Pose to Transfromation matrix --------------------- #


def pose_vector_to_se3(pose):
    """
    Convert a 6D pose vector [x, y, z, rx, ry, rz] to a 4x4 SE(3) transformation matrix.
    """
    # Extract translation and rotation vector
    x, y, z, rx, ry, rz = pose
    t = np.array([x, y, z])  # Translation vector

    # Compute rotation matrix from axis-angle
    rotation_vector = np.array([rx, ry, rz])
    theta = np.linalg.norm(rotation_vector)

    if theta < 1e-10:  # Avoid division by zero (no rotation case)
        R = np.eye(3)
    else:
        R = Rotation.from_rotvec(rotation_vector).as_matrix()

    # Construct SE(3) matrix
    T = np.eye(4)
    T[:3, :3] = R  # Insert rotation matrix
    T[:3, 3] = t   # Insert translation

    return T
# ------------------------- Transformation to UR Pose ------------------------ #


def transformation_matrix_to_pose(matrix):
    """
    Converts a 4x4 homogeneous transformation matrix to a 6D pose vector (position + axis-angle rotation).
    Parameters:
        matrix (np.array): 4x4 homogeneous transformation matrix
    Returns:
        np.array: 6D pose vector (x, y, z, rx, ry, rz)
    """
    position = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    axis_angle = rotation_matrix_to_axis_angle(rotation_matrix)
    pose = np.concatenate([position, axis_angle])

    return pose

# -------------------------- Rotation to Axis Angle -------------------------- #


def rotation_matrix_to_axis_angle(rotation_matrix):
    """
    Converts a 3x3 rotation matrix to an axis-angle rotation vector.
    Parameters:
        rotation_matrix (np.array): 3x3 rotation matrix
    Returns:
        np.array: Axis-angle rotation vector (rx, ry, rz)
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


# ------------------------------- Move to start ------------------------------ #


def move_to_start_configuration(robot_control, Q0, P0, axis_angle):
    input("Press enter to move to start configuration.")
    robot_control.moveJ(Q0)
    robot_control.moveL(P0.t.tolist() + axis_angle)


def move_robot(robot_control, robot_receive, T1):
    # Parameters
    velocity = 0.5
    acceleration = 0.5
    lookahead_time = 0.1
    gain = 300

    dt = 0.008
    T = 2
    t = np.arange(0, T, dt)

    T0 = robot_receive.getActualTCPPose()
    P0 = T0[:3]
    P1 = T1.t

    axis_angle = rotation_matrix_to_axis_angle(T1.R)
    axis_angle = [2.467, -1.767, 0.633]

    delta2 = P1 - P0

    T_poly_x = rtb.tools.trajectory.quintic(P0[0], P1[0], t, qdf=delta2[0])
    T_poly_y = rtb.tools.trajectory.quintic(P0[1], P1[1], t, qdf=delta2[1])
    T_poly_z = rtb.tools.trajectory.quintic(P0[2], P1[2], t, qdf=delta2[2])

    Q0 = robot_receive.getActualQ()
    T0 = SE3(pose_vector_to_se3(T0))
    # move_to_start_configuration(robot_control, Q0, T0, axis_angle)
    # robot_control.moveJ(Q0)

    T_trap_1 = rtb.tools.trajectory.ctraj(T0, T1, t)
    for i in range(0, len(T_trap_1)):
        # pi = [T_poly_x.s[i], T_poly_y.s[i], T_poly_z.s[i]] + axis_angle
        # robot_control.servoL(pi, velocity, acceleration, dt, lookahead_time, gain)
        pi = T_trap_1.t[i].tolist() + axis_angle

        robot_control.servoL(pi, velocity, acceleration,
                             dt, lookahead_time, gain)
        time.sleep(dt)
    # ----------------------------------- test ----------------------------------- #
      # Joint configurations
    Q0 = np.array(
        [
            0.6743066906929016,
            -1.3727410475360315,
            -1.204346005116598,
            4.2224321365356445,
            1.159425139427185,
            0.3307466506958008,
        ]
    )

    P0 = SE3(0.36177, 0.10525, 0.64767)
    P1 = SE3(0.36177, -0.22975, 0.3162)
    T_trap_1 = rtb.tools.trajectory.ctraj(P0, P1, t)
    move_to_start_configuration(robot_control, Q0, P0, axis_angle)
    input("Press enter to start Cartesian-space trapezoidal velocity profile")
    for i in range(0, len(T_trap_1)):
        pi = T_trap_1.t[i].tolist() + axis_angle
        robot_control.servoL(pi, velocity, acceleration,
                             dt, lookahead_time, gain)
        time.sleep(dt)
