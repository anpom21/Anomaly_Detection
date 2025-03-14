import math
import rtde_control
import rtde_receive
import argparse
import roboticstoolbox as rtb
from mpl_toolkits.mplot3d import Axes3D
from rtde_control import RTDEControlInterface
# Recieve interface
from rtde_receive import RTDEReceiveInterface
import numpy as np
from spatialmath import SO3
import spatialmath as sm

def move_to_start_configuration(robot_control, Q0=0, P0=0, axis_angle=0):
    axis_angle = [2.467, -1.767, 0.633]
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

    input("Press enter to move to start configuration.")
    robot_control.moveJ(Q0)
    robot_control.moveL(P0.t.tolist() + axis_angle)

#!/usr/bin/env python3

def calculate_light_positions(n_light, center, height, distance):
    """
    Calculate positions for light sources in a circle around a center point using spherical coordinates.
    
    Args:
        n_light: Number of light sources
        center: Center coordinate [x, y, z]
        height: Height of light sources (z-coordinate)
        distance: Distance from center to each light source in the xy-plane
    
    Returns:
        List of positions [x, y, z, rx, ry, rz] for each light source
    """
    positions = []
    
    # Calculate the angle between each light source (azimuthal angle in spherical coordinates)
    angle_increment = 2 * math.pi / n_light
    
    for i in range(n_light):
        # Calculate the angle for this light source
        phi = i * angle_increment
        
        r = distance
        theta = np.arccos(height/r)
        
        # Calculate position (using spherical coordinates converted to cartesian)
        # In this case, all points are at same distance (r) and height (theta is fixed)
        x = center[0] + r * np.sin(theta) * np.cos(phi)
        y = center[1] + r * np.sin(theta) * np.sin(phi)
        z = center[2] + r * np.cos(theta)

        positions.append([x, y, z])
    
    return positions

def position_to_pose(position, center):
    # The pose should be represented as a 4x4 transformation matrix
    z_axis = np.array(center) - np.array(position)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # Define the x-axis as the tangent to the circle
    x_axis = np.array([-z_axis[1], z_axis[0], 0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Define the y-axis as the cross product of z and x
    y_axis = np.cross(z_axis, x_axis)
    
    
    # Create the rotation matrix
    rotation_matrix = np.array([[x_axis[0], y_axis[0], z_axis[0]],
                                 [x_axis[1], y_axis[1], z_axis[1]],
                                 [x_axis[2], y_axis[2], z_axis[2]]])
    
    # Create the transformation matrix
    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix
    pose[:3, 3] = position
    
    return pose 
    
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

def rotation_matrix_to_axis_angle(rotation_matrix):
    """
    Converts a 3x3 rotation matrix to an axis-angle rotation vector.
    Parameters:
        rotation_matrix (np.array): 3x3 rotation matrix
    Returns:
        np.array: Axis-angle rotation vector (rx, ry, rz)
    """
    assert np.allclose(np.dot(rotation_matrix, rotation_matrix.T), np.eye(3)), "Input is not a valid rotation matrix"
    
    angle = np.arccos((np.trace(rotation_matrix) - 1) / 2)
    
    if angle == 0:
        return np.array([0, 0, 0])
    elif angle == np.pi:
        x = np.sqrt((rotation_matrix[0, 0] + 1) / 2)
        y = np.sqrt((rotation_matrix[1, 1] + 1) / 2) * np.sign(rotation_matrix[0, 1])
        z = np.sqrt((rotation_matrix[2, 2] + 1) / 2) * np.sign(rotation_matrix[0, 2])
        return np.array([x, y, z]) * angle
    else:
        rx = rotation_matrix[2, 1] - rotation_matrix[1, 2]
        ry = rotation_matrix[0, 2] - rotation_matrix[2, 0]
        rz = rotation_matrix[1, 0] - rotation_matrix[0, 1]
        axis = np.array([rx, ry, rz]) / (2 * np.sin(angle))
        return axis * angle

def move_robot_to_positions(poses, robot_ip, velocity=0.2, acceleration=1.4):
    """
    Send positions to UR5 robot and move to each position using moveJ.
    
    Args:
        positions: List of positions [x, y, z, rx, ry, rz] for each light source
        robot_ip: IP address of the UR5 robot
        velocity: Joint velocity of robot (rad/s)
        acceleration: Joint acceleration of robot (rad/s^2)
    """
    # Connect to the robot
    rtde_c = RTDEControlInterface(robot_ip)
    # rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    
    # current_q = rtde_r.getActualQ()
    
    # print("Current joint positions:", current_q)
    
    # Check if the connection is established
    if not rtde_c.isConnected():
        print("Failed to connect to the robot")
        return False
    
    # Move to each position
    for i, T in enumerate(poses):
        print(f"Move to position {i+1}/{len(poses)}: {T}")
        
        
        # Wait for input before moving to the next position
        input("Press enter to move to the next position")
        
        # Extract the position and orientation from the transformation matrix
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        P0 = sm.SE3(position)
        move_to_start_configuration(rtde_c, P0 = P0)
        
        # # Convert to axis angle representation
        # rotation_vector = transformation_matrix_to_pose(T)
        
        # # Combine position and rotation vector into a pose
        # pose = np.concatenate((position, rotation_vector))
        
        # # Print current pose of the robot
        # print("Current pose:", rtde_c.getActualTCPPose())

        # # Perform inverse kinematics to get the joint positions
        # joint_positions = rtde_c.getInverseKinematics(pose)

        # print("Joint positions:", joint_positions)
        
        # # Use moveJ to move to the position
        # success = rtde_c.moveJ(joint_positions)
        
        # if not success:
        #     print(f"Failed to move to position {i+1}")
        #     rtde_c.disconnect()
        #     return False
    
    # Disconnect from the robot
    rtde_c.disconnect()
    print("Movement completed successfully")
    return True

def main():
    
    # Parameters
    center = [0.2, 0.2, 0.2] #0.3
    light_radius = 0.4
    light_height = 0.3
    n_light = 5
    
    # Calculate light positions
    positions = calculate_light_positions(n_light, center, light_height, light_radius)
    
    # Plot the positions
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates from positions
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]
    z_coords = [pos[2] for pos in positions]
    
    # Plot light source positions
    ax.scatter(x_coords, y_coords, z_coords, c='red', marker='o', s=100, label='Light Sources')
    
    # Plot the center point
    ax.scatter(center[0], center[1], center[2], c='blue', marker='o', s=200, label='Center')
    
    # Plot a half-sphere (barely visible)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi / 2, 100)
    x_sphere = light_radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y_sphere = light_radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z_sphere = light_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1)
    
    # Get pose for each light source
    poses = [position_to_pose(pos, center) for pos in positions]
    
    # Plot the light source as frames
    for i, pose in enumerate(poses):
        # Extract the position and orientation
        position = pose[:3, 3]
        orientation = pose[:3, :3]
        
        # Define the frame size
        frame_size = 0.1
        
        # Define the frame axes
        x_axis = frame_size * orientation[:, 0] + position
        y_axis = frame_size * orientation[:, 1] + position
        z_axis = frame_size * orientation[:, 2] + position
        
        # Plot the frame
        ax.plot([position[0], x_axis[0]], [position[1], x_axis[1]], [position[2], x_axis[2]], c='red')
        ax.plot([position[0], y_axis[0]], [position[1], y_axis[1]], [position[2], y_axis[2]], c='green')
        ax.plot([position[0], z_axis[0]], [position[1], z_axis[1]], [position[2], z_axis[2]], c='blue')
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{n_light} light position\n Center: {center} \nLight Height: {light_height} \nRadius: {light_radius}')
    
    # Max x y z values
    max_val = max(max(x_coords), max(y_coords), max(z_coords))
    # Min x y z values
    min_val = min(min(x_coords), min(y_coords), min(z_coords))
    
    # Set equal scaling for all axes
    # ax.set_xlim([min(x_coords + [center[0]]) - 0.3, max(x_coords + [center[0]]) + 0.3])
    # ax.set_ylim([min(y_coords + [center[1]]) - 0.3, max(y_coords + [center[1]]) + 0.3])
    # ax.set_zlim([min(z_coords + [center[2]]) - 0.1, max(z_coords + [center[2]]) + 0.1])
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 3/4])
    # Set the aspect ratio to be equal
    ax.set_aspect('auto')
    ax.legend()
    plt.show()
    
    # Ask for confirmation before moving the robot
    response = input("Do you want to send these positions to the robot? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled")
        return
    
    ip = "192.168.1.30"
    
    # Send positions to the robot
    move_robot_to_positions(poses, ip)
    
        





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
    