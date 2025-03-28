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
# from kinematic_functions import position_to_pose, pose_vector_to_se3, rotation_matrix_to_axis_angle, student_IK, move_to_start_configuration

#!/usr/bin/env python3

# --------------------- Postions to Transformation matrix -------------------- #


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

    return SE3.Rt(rotation_matrix, position)


def half_sphere_simple(n_light, center, height, distance, lighting_angle):
    """
    Calculate positions for light sources in a circle around a center point using spherical coordinates.
    # Link to spherical coordinates
    # https://www.researchgate.net/figure/Figure-A1-Spherical-coordinates_fig8_284609648


    Args:
        n_light (Int): Number of light sources
        center: Center coordinate [x, y, z]
        height (Float): Height of light sources (z-coordinate)
        distance (Float): Distance from center to each light source in the xy-plane

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
        theta = np.deg2rad(90 - lighting_angle)

        # Calculate position (using spherical coordinates converted to cartesian)
        # In this case, all points are at same distance (r) and height (theta is fixed)
        x = center[0] + r * np.sin(theta) * np.cos(phi)
        y = center[1] + r * np.sin(theta) * np.sin(phi)
        z = center[2] + r * np.cos(theta)

        positions.append([x, y, z])

    return positions


# ---------------------------- Iterative Repulsion --------------------------- #


def random_on_sphere(num_points, radius=1.0):
    # Returns Nx3 points on the surface of a unit sphere
    pts = np.random.normal(size=(num_points, 3))
    pts /= np.linalg.norm(pts, axis=1)[:, None]
    # Scale to the desired radius
    pts *= radius

    # Enforce z >= 0: if z < 0, reflect it to z > 0, then re-scale
    for i in range(num_points):
        if pts[i, 2] < 0:
            # Flip z
            pts[i, 2] = -pts[i, 2]
            # Re-project to radius R
            norm = np.linalg.norm(pts[i])
            if norm > 1e-9:
                pts[i] *= (radius / norm)
    return pts


def repulsion_step(positions, step_size=0.01, radius=1.0):
    """
    One iteration of the repulsion step on the *positive hemisphere*. 
    Each point is repelled from the others, then projected onto z >= 0,
    then clamped to the sphere of radius `radius`.
    """
    new_positions = np.copy(positions)
    num_points = len(positions)

    for i in range(num_points):
        force = np.zeros(3)
        for j in range(num_points):
            if i == j:
                continue
            diff = positions[i] - positions[j]
            dist_sq = np.dot(diff, diff) + 1e-9  # small offset to avoid /0
            # Simple "inverse-square" repulsion
            force += diff / dist_sq

        # Update position
        new_positions[i] += step_size * force

        # Project to sphere of radius R
        norm = np.linalg.norm(new_positions[i])
        if norm > 1e-9:
            new_positions[i] *= (radius / norm)

        # If z < 0, reflect to ensure z >= 0, then re-project to radius R
        if new_positions[i, 2] < 0:
            new_positions[i, 2] = -new_positions[i, 2]
            norm = np.linalg.norm(new_positions[i])
            if norm > 1e-9:
                new_positions[i] *= (radius / norm)

    return new_positions


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


def plot_simple(T, center, light_radius, light_height, n_light):
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

    # ax.plot_surface(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1)

    plt.axis('scaled')
    plt.axis('equal')
    plt.axis('square')
    ax.set_aspect('equal', adjustable='box')

    # Set the aspect ratio to be equal
    # ax.set_aspect('auto')
    ax.legend()
    plt.show()


def main():

    # Example usage:

    # Parameters
    n_light = 4
    center = [0.379296019468532, -
              0.4164582402752736, -0.086414021169025]  # 0.3
    light_radius = 0.2
    light_height = 0.18

    # ---- Calculate light positions ---- #

    # # # Half Sphere # # #
    pos_half_sphere = half_sphere_simple(
        n_light, center, light_height, light_radius, 60)

    # # # Repulsion # # #
    pos_itr = random_on_sphere(n_light)
    for iteration in range(1000):
        pos_itr = repulsion_step(pos_itr, step_size=0.01)

    # --- Position to pose transformation ---- #
    T_light = [position_to_pose(pos, center) for pos in pos_half_sphere]

    # --- Plotting --- #
    plot_simple(T_light, center, light_radius, light_height, n_light)
    plot_light_positions(T_light, center, light_radius, light_height, n_light)


if __name__ == "__main__":
    main()
