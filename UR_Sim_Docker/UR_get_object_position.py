from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import yaml


def load_yaml_position(filename):
    """
    Load a YAML file and return the data.
    """
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    # Extract the position from the data
    position = data.get('position', None)
    if position is None:
        raise ValueError("Position not found in the YAML file.")
    return position


def main():
    # ------------------------ Connect to the robot ------------------------ #
    # Replace with your robot's IP address
    ip = "localhost"
    ip = "192.168.1.30"
    # Connect to the robot
    rtde_c = RTDEControlInterface(ip)
    rtde_r = RTDEReceiveInterface(ip)
    # Check connection
    if rtde_c.isConnected():
        print("[INFO] Connected to the robot")
    else:
        print("[ERROR] Failed to connect to the robot")
        return

    current_q = rtde_r.getActualQ()
    print("[INFO] Current joint angles:", current_q)
    # Pose of the end effector in the base frame
    current_pose = rtde_r.getActualTCPPose()
    print("[INFO] Current end effector pose:", current_pose)
    # Move to q
    q_rtb = [-2.314551, - 0.20553135,  1.71054773,
             2.29812167,  0.79435806, - 0.67166277]
    q = [0.51193746, -0.87995084,  0.45437693,
         0.74222859,  0.956926,    0.21843418]
    # rtde_c.moveJ(q, 0.2, 0.2)
    q_sim = rtde_r.getActualQ()

    # -------------------------- Set robot to freedrive -------------------------- #
    rtde_c.teachMode()
    print("[INFO] Robot in freedrive mode. Press enter to continue...")
    input()

    # End teach mode
    rtde_c.endTeachMode()
    print("[INFO] Robot in normal mode.")

    position = rtde_r.getActualTCPPose()[:3]

    print("Position:", rtde_r.getActualTCPPose()[:3])
    print("Pose:", rtde_r.getActualTCPPose())
    print("Joint angles:", rtde_r.getActualQ())

    # ------------------------ Save position in yaml file ------------------------ #
    filename = "metal_plate_position.yaml"

    save_yaml = input("[PROMPT] Save the position in a YAML file? (y/n)")

    if save_yaml.lower() != 'y':
        print("[INFO] Exiting without saving.")
        return

    data = {"position": position}

    with open(filename, "w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

    print("[INFO] Position saved in metal_plate_position.yaml")

    # Check it was saved correctly
    position_loaded = load_yaml_position(filename)
    print("[INFO] Position loaded from YAML file:", position_loaded)


if __name__ == "__main__":
    main()
