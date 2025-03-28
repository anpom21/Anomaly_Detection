import math
# import roboticstoolbox as rtb
from mpl_toolkits.mplot3d import Axes3D
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from scipy.spatial.transform import Rotation
import numpy as np
from spatialmath import SE3
import spatialmath as sm
import yaml
import matplotlib.pyplot as plt
import pickle
# from kinematic_functions import position_to_pose, pose_vector_to_se3, rotation_matrix_to_axis_angle, student_IK, move_to_start_configuration

#!/usr/bin/env python3


# ----------------------------- Activate UR robot ---------------------------- #
def activate_robot():
    ip = "192.168.1.30"
    ur_control = RTDEControlInterface(ip)
    ur_receive = RTDEReceiveInterface(ip)
    return ur_control, ur_receive


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


def set_equal_3d(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    max_range = max(x_limits[1] - x_limits[0],
                    y_limits[1] - y_limits[0],
                    z_limits[1] - z_limits[0])
    mid_x = 0.5 * (x_limits[0] + x_limits[1])
    mid_y = 0.5 * (y_limits[0] + y_limits[1])
    mid_z = 0.5 * (z_limits[0] + z_limits[1])
    ax.set_xlim3d([mid_x - 0.5*max_range, mid_x + 0.5*max_range])
    ax.set_ylim3d([mid_y - 0.5*max_range, mid_y + 0.5*max_range])
    ax.set_zlim3d([mid_z - 0.5*max_range, mid_z + 0.5*max_range])

# --------------------------- Save and load to YAML -------------------------- #


def yaml_save_positions(positions, filename='positions.yaml'):
    positions_list = [pos.tolist() for pos in positions]
    print(positions_list)
    data = {
        'positions': positions_list
    }
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    print("Positions saved to positions.yaml")


def yaml_load_positions(filename='positions.yaml'):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
        positions_list = data['positions']
        positions = [np.array(pos) for pos in positions_list]

    print("Positions loaded from positions.yaml")
    return positions

# -------------------------- Plot lighting positions ------------------------- #


def plot_light_positions(T, center, light_radius, light_height, n_light):
    """
    Plot the positions of light sources and the center point in 3D space.
    Clicking on a yellow light source will change its color to green.
    """
    positions = [t.t for t in T]
    # Create initial color list for the light sources (all yellow)
    global colors  # used in the onpick callback
    colors = ['yellow'] * len(positions)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]
    z_coords = [pos[2] for pos in positions]

    # Create a scatter plot with picking enabled
    # store reference for the onpick callback
    global scat, scat_scenter, pos_light, pos_intermediate, robot_pos
    scat = ax.scatter(x_coords, y_coords, z_coords, c=colors,
                      marker='o', s=100, label='Light Sources', picker=True)
    pos_light = []
    pos_intermediate = []
    robot_pos = robot_positions(center)

    # Plot frames for each light source (unchanged)
    for i, pose in enumerate(T):
        position = pose.t
        orientation = pose.R
        frame_size = 0.1
        x_axis = frame_size * orientation[:, 0] + position
        y_axis = frame_size * orientation[:, 1] + position
        z_axis = frame_size * orientation[:, 2] + position
        ax.plot([position[0], x_axis[0]], [position[1], x_axis[1]],
                [position[2], x_axis[2]], c='red')
        ax.plot([position[0], y_axis[0]], [position[1], y_axis[1]],
                [position[2], y_axis[2]], c='green')
        ax.plot([position[0], z_axis[0]], [position[1], z_axis[1]],
                [position[2], z_axis[2]], c='blue')

    # Plot the center point
    scat_scenter = ax.scatter(center[0], center[1], center[2], c='blue',
                              marker='o', s=200, label='Center', picker=True)

    # Plot a half-sphere (for visualization)
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi / 2, 10)
    x_sphere = light_radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y_sphere = light_radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z_sphere = light_radius * \
        np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x_sphere, y_sphere, z_sphere,
                    color='gray', alpha=0.1, rstride=1, cstride=1, linewidth=0)

    # Plot a UR Base (cylinder)
    cyl_r = 0.08
    z_cylinder = np.linspace(0, light_height, 3)
    theta_cylinder = np.linspace(0, 2 * np.pi, 10)
    theta_cylinder, z_cylinder = np.meshgrid(theta_cylinder, z_cylinder)
    x_cylinder = cyl_r * np.cos(theta_cylinder)
    y_cylinder = cyl_r * np.sin(theta_cylinder)
    ax.plot_surface(x_cylinder, y_cylinder, z_cylinder,
                    color='blue', alpha=0.3)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(
        f'{n_light} light position\n Center: {center} \nLight Height: {light_height} \nRadius: {light_radius}')

    set_equal_3d(ax)
    ax.legend()

    # --- onpick event callback ---
    def onpick(event):
        global scat_scenter, colors, pos_light
        # Check if the picked artist is our scatter plot
        if event.artist == scat:
            # Change color of the clicked point to green
            for i in event.ind:
                colors[i] = 'green'
            scat.set_color(colors)

            # Add the clicked position to the order list
            position_clicked = positions[event.ind[0]]
            robot_pos.add_light_position(position_clicked)

            ax.text(position_clicked[0], position_clicked[1],
                    position_clicked[2], str(robot_pos.count), fontsize=12)
        elif event.artist == scat_scenter:
            # Reset the position order
            print('Position order was reset')
            robot_pos.clear()
            colors = ['yellow'] * len(positions)
            scat.set_color(colors)
            # Remove text labels
            for txt in ax.texts:
                txt.set_visible(False)

        plt.draw()
    # -- on key press event callback -- #

    def on_key_press(event):
        if event.key == "enter":
            print("Make intermediate pose.")
            ur_control, ur_receive = activate_robot()

            if ur_control.teachMode():
                print("[INFO] Teach mode activated.")
                print("[INFO] Configure robot to desired intermediate position.")
                print("[INFO] Press 'Space' to SAVE the position.")
                print("[INFO] Press 'Enter' to EXIT teach mode.")
            else:
                print("[ERROR] Teach mode could not be activated.")

            input("Press enter to save position and exit teach mode.")

            pose = ur_receive.getActualTCPPose()
            robot_pos.add_intermediate(pose)
            ax.text(pose[0], pose[1], pose[2], str(
                robot_pos.count), fontsize=12)
            print("Position saved.")
            print(pose[:3])
            ur_control.endTeachMode()
            print("[INFO] Teach mode deactivated.")

    # Connect the onpick event handler
    fig.canvas.mpl_connect('pick_event', onpick)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    plt.show()

    save_pos = input("\nDo you want to save the positions? (y/n): ")
    filename = "robot_position.pkl"
    if save_pos == "y":
        file = open(filename, "wb")
        pickle.dump(robot_pos, file)
        file.close()
        print("Positions saved to ", filename)

    file = open(filename, "rb")
    robot_pos = pickle.load(file)
    file.close()

    print("Positions loaded from ", filename)
    print("Positions", robot_pos.positions)
    print("Number of positions: ", robot_pos.count)
    print("Position labels: ", robot_pos.labels)
    print("Number of light positions: ", robot_pos.light_positions)
    print("Number of intermediate positions: ",
          robot_pos.intermediate_positions)

    return robot_pos


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

    # Plot a UR Base
    cyl_r = 0.08
    z_cylinder = np.linspace(0, light_height, 3)
    theta_cylinder = np.linspace(0, 2 * np.pi, 10)
    theta_cylinder, z_cylinder = np.meshgrid(theta_cylinder, z_cylinder)
    x_cylinder = cyl_r * np.cos(theta_cylinder)
    y_cylinder = cyl_r * np.sin(theta_cylinder)

    ax.plot_surface(x_cylinder, y_cylinder, z_cylinder,
                    color='blue', alpha=0.3)

    set_equal_3d(ax)

    # Set the aspect ratio to be equal
    # ax.set_aspect('auto')
    ax.legend()
    plt.show()
# ---------------------------------------------------------------------------- #
#                                    CLASSES                                   #
# ---------------------------------------------------------------------------- #


class robot_positions:
    def __init__(self, object_position):
        self.light_positions = []
        self.intermediate_positions = []
        self.positions = []
        self.labels = []
        self.count = 0
        self.object_position = object_position

    def add_light_position(self, position):
        self.light_positions.append(position)
        self.positions.append(position)
        self.labels.append("Light")
        self.count += 1

    def add_intermediate_position(self, position):
        self.intermediate_positions.append(position)
        self.positions.append(position)
        self.labels.append("Intermediate")
        self.count += 1

    def clear(self):
        self.light_positions.clear()
        self.intermediate_positions.clear()
        self.positions.clear()
        self.count = 0
# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #


def main():

    # Example usage:

    # Parameters
    n_light = 4
    center = [0.379296019468532, -
              0.4164582402752736, -0.086414021169025]  # 0.3
    light_radius = 0.2
    light_height = 0.18
    light_angle = 60

    # ---- Calculate light positions ---- #

    # # # Half Sphere # # #
    pos_half_sphere = half_sphere_simple(
        n_light, center, light_height, light_radius, light_angle)

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
