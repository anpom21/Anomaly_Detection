import yaml
import pickle
from photometric_positions import robot_positions

# PARAMETERS
rob_pos_filename = "robot_position.pkl"
dataset_path = "irl_dataset/"

# Connect to the robot
ip_real = "192.168.1.30"
ip_sim = "localhost"

# ---- #
# ---------------------------- Load the positions ---------------------------- #
file = open(rob_pos_filename, "rb")
robot_position = pickle.load(file)
file.close()
print("[INFO] Positions loaded successfully")
