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
from kinematic_functions import position_to_pose, pose_vector_to_se3, rotation_matrix_to_axis_angle, student_IK, move_to_start_configuration
from photometric_positions import calculate_light_positions
#!/usr/bin/env python3


def auto_home_position(robot_r, center):
    """Automatically calculate the home position for the robot based on the center point.

    Args:
        robot_r (RTDEReceiveInterface): RTDE Recieve object for the robot
        center (np.array): Position of the center point

    Returns:
        q_home (np.array): Home joint position for the robot
    """
    q = robot_r.getActualQ()
    angle1 = np.arctan2(center[1], center[0]) + np.deg2rad(-165)
    angle2 = np.arctan2(center[1], center[0]) + np.deg2rad(195)
    angles = [angle1, angle2]
    min_index = np.argmin([np.abs(q[0] - angle1), np.abs(q[0] - angle2)])
    center_angle = angles[min_index]
    q_home = [center_angle, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0]

    return q_home

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


def move_robot_to_positions(transformation_matrix, robot_ip, center, velocity=1.05, acceleration=1.4):
    """
    Send positions to UR5 robot and move to each position using moveJ.

    Args:
        positions: List of positions [x, y, z, rx, ry, rz] for each light source
        robot_ip: IP address of the UR5 robot
        velocity: Joint velocity of robot (rad/s)
        acceleration: Joint acceleration of robot (rad/s^2)
    """
    # Connect to the robot
    ur_control = RTDEControlInterface(robot_ip)
    ur_receive = RTDEReceiveInterface(robot_ip)

    custom_q_home = [-0.8106325308429163, -2.042311970387594, -1.8650391737567347,
                     5.403944969177246, 1.4234832525253296, -1.0592706839190882]

    # Check if the connection is established
    if not ur_control.isConnected():
        print("Failed to connect to the robot")
        return False

    # Move to each position
    for i, T in enumerate(transformation_matrix):
        print(f"Move to position {i+1}/{len(transformation_matrix)}: \n{T.t}")
        pose = list(T.t.tolist()) + list(rotation_matrix_to_axis_angle(T.R))

        input("\nPress to home")

        success = ur_control.moveJ(
            custom_q_home, speed=velocity, acceleration=acceleration)

        input("\nPress enter to move to the next position")
        success = ur_control.moveJ_IK(
            pose, speed=velocity, acceleration=acceleration)

        # move_robot(rtde_c, rtde_r, T)
        if not success:
            print(f"Failed to move to position {i+1}")
            ur_control.disconnect()
            return False

    # Disconnect from the robot
    ur_control.disconnect()
    print("Movement completed successfully")
    return True


def plot_light_positions(T, center, light_radius, light_height, n_light):
    """
    Plot the positions of light sources and the center point in 3D space.

    Args:
        positions: List of positions [x, y, z] for each light source
        center: Center coordinate [x, y, z]
        light_radius: Radius of the circle in the xy-plane
        light_height: Height of light sources (z-coordinate)
        n_light: Number of light sources
    """
    positions = [t.t for t in T]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates from positions
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]
    z_coords = [pos[2] for pos in positions]

    # Plot light source positions
    ax.scatter(x_coords, y_coords, z_coords, c='yellow',
               marker='o', s=100, label='Light Sources')

    # Plot the center point
    ax.scatter(center[0], center[1], center[2], c='blue',
               marker='o', s=200, label='Center')

    # Plot a half-sphere (barely visible)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi / 2, 100)
    x_sphere = light_radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y_sphere = light_radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z_sphere = light_radius * \
        np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1)

    # Get pose for each light source

    # Plot the light source as frames
    for i, pose in enumerate(T):
        # Extract the position and orientation
        position = pose.t
        orientation = pose.R

        # Define the frame size
        frame_size = 0.1

        # Define the frame axes
        x_axis = frame_size * orientation[:, 0] + position
        y_axis = frame_size * orientation[:, 1] + position
        z_axis = frame_size * orientation[:, 2] + position

        # Plot the frame
        ax.plot([position[0], x_axis[0]], [position[1], x_axis[1]],
                [position[2], x_axis[2]], c='red')
        ax.plot([position[0], y_axis[0]], [position[1], y_axis[1]],
                [position[2], y_axis[2]], c='green')
        ax.plot([position[0], z_axis[0]], [position[1], z_axis[1]],
                [position[2], z_axis[2]], c='blue')

    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(
        f'{n_light} light position\n Center: {center} \nLight Height: {light_height} \nRadius: {light_radius}')

    # Max x y z values
    max_val = max(max(x_coords), max(y_coords), max(z_coords))
    # Min x y z values
    min_val = min(min(x_coords), min(y_coords), min(z_coords))
    print(max_val)
    print(min_val)

    # Plot a cylinder
    cyl_r = 0.08
    z_cylinder = np.linspace(0, light_height, 100)
    theta_cylinder = np.linspace(0, 2 * np.pi, 100)
    theta_cylinder, z_cylinder = np.meshgrid(theta_cylinder, z_cylinder)
    x_cylinder = cyl_r * np.cos(theta_cylinder)
    y_cylinder = cyl_r * np.sin(theta_cylinder)

    ax.plot_surface(x_cylinder, y_cylinder, z_cylinder,
                    color='blue', alpha=0.3)

    # Set same axes limits
    # Calculate axes center for each axis
    x_mid = (max(x_coords) + min(x_coords)) / 2
    y_mid = (max(y_coords) + min(y_coords)) / 2
    z_mid = (max(z_coords) + min(z_coords)) / 2

    # Set axes limits

    # ax.set_xlim([min_val, max_val])
    # ax.set_ylim([min_val, max_val])
    # ax.set_zlim([min_val, max_val])
    # Equal aspect ratio
    # ax.set_box_aspect([1, 1, 1])
    plt.axis('scaled')
    plt.axis('equal')
    plt.axis('square')
    ax.set_aspect('equal', adjustable='box')

    # Set the aspect ratio to be equal
    # ax.set_aspect('auto')
    ax.legend()
    plt.show()


def main():

    # Parameters
    center = [0.379296019468532, -
              0.4164582402752736, -0.086414021169025]  # 0.3
    light_radius = 0.3
    light_height = 0.27
    n_light = 4

    # Calculate light positions
    positions = calculate_light_positions(
        n_light, center, light_height, light_radius)

    T_light = [position_to_pose(pos, center) for pos in positions]

    plot_light_positions(T_light, center, light_radius, light_height, n_light)

    # Ask for confirmation before moving the robot
    # response = input("Do you want to send these positions to the robot? (y/n): ")
    # if response.lower() != 'y':
    #    print("Operation cancelled")
    #    return

    ip = "192.168.1.30"

    # Send positions to the robot
    move_robot_to_positions(T_light, ip, center)


"""
def main():
    ip = "192.168.1.29"
    
    
    parser = argparse.ArgumentParser(description='Control UR5 robot to move around light sources')
    parser.add_argument('--n_light', type=int, default=8, help='Number of light sources')
    parser.add_argument('--center_x', type=float, default=0.0, help='X coordinate of center point')
    parser.add_argument('--center_y', type=float, default=0.0, help='Y coordinate of center point')
    parser.add_argument('--center_z', type=float, default=0.3, help='Z coordinate of center point')
    parser.add_argument('--height', type=float, default=0.5, help='Height of light sources')
    parser.add_argument('--distance', type=float, default=0.3, help='Distance from center to each light source')
    parser.add_argument('--robot_ip', type=str, default=ip, help='IP address of the UR5 robot')
    
    args = parser.parse_args()
    
    # Parameters
    center = [args.center_x, args.center_y, args.center_z]
    
    # Calculate light positions
    positions = calculate_light_positions(args.n_light, center, args.height, args.distance)
    
    # Print the positions
    for i, pos in enumerate(positions):
        print(f"Light {i+1} position: {pos}")
    
    # Ask for confirmation before moving the robot
    response = input("Do you want to send these positions to the robot? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled")
        return
    
    # Send positions to the robot
    move_robot_to_positions(positions, args.robot_ip)
"""
if __name__ == "__main__":
    main()
