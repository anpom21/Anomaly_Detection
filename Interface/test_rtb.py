from roboticstoolbox import models
import spatialmath as sm
import numpy as np

# Load the Puma560 robot model
robot = models.DH.Puma560()

print(f"Robot loaded: {robot.name}")
print(f"Number of joints: {robot.n}")

# Define a test joint configuration (middle of joint range)
q = (robot.qr + robot.qz) / 2
print(f"Test joint configuration: {q}")

# Compute forward kinematics
T = robot.fkine(q)
print("Forward kinematics result (end-effector pose):")
print(T)

# Compute inverse kinematics to recover q from T
sol = robot.ikine_LM(T)
print("Inverse kinematics solution:")
print(sol)

# Optionally plot if you're in a Jupyter Notebook or have matplotlib
try:
    robot.plot(q, block=True)
except Exception as e:
    print(f"Plotting skipped or failed: {e}")
