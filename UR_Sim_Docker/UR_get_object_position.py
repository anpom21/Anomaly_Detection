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
    # ip = "localhost"
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

    input("[PROMPT] Save the position in a YAML file? (y/n)")

    if input().lower() != 'y':
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
