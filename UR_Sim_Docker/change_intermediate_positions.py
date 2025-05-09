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

# Save the positions to pickle file
with open("robot_position.pkl", "wb") as file:
    pickle.dump(robot_position, file)
[-5.553293172513143, 0.11027169227600098, 1.1755104064941406,
    0.6471174955368042, -0.5075510183917444, 1.161909580230713]
print("[INFO] Positions loaded successfully")
