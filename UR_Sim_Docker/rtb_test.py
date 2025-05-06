import roboticstoolbox as rtb
import spatialmath as sm
from spatialmath import base
import numpy as np
import matplotlib.pyplot as plt

# Load UR5 model
robot = rtb.models.UR5()

# Target end-effector pose
# T = sm.SE3(0.5, -0.2, 0.3) * sm.SE3.Ry(np.pi)
# print("T:", T)

goal_q = [0.51193746, -0.8799508400000002, 0.45437693000000035,
          0.74222859, 0.9569260000000002, 0.21843418000000003]
goal_T = [-0.49448664610934034, -0.4965673393378531, 0.5318728837627703,
          1.67455105207058, -0.7335665748267906, -0.011121444046847614]

# Check thath pose matches the joint angles
T = robot.fkine(goal_q)
R_dh_to_urdf = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]
])
T_correction = np.eye(4)
T_correction[:3, :3] = R_dh_to_urdf
T_rtb_corrected = robot.fkine(goal_q).A @ T_correction
print("FK of joint angles:", T_rtb_corrected)
print("FK of non corrected joint angles:", T.A)
print("Goal_T:", goal_T)
v = goal_T[3:] / np.linalg.norm(goal_T[3:])
theta = np.linalg.norm(goal_T[3:])
print("v:", v)
print("theta:", theta)
R = sm.base.angvec2r(theta, v, unit='rad')
t = np.array(goal_T[:3])
print("t:", t)
print("R:", R)
T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = t
T = sm.SE3(T)
print("T:", T)
# Convert to SE3 object
# T_tool = np.eye(4)
# T_tool[:3, 3] = [0, 0, 0.185]

# T = T @ np.linalg.inv(T_tool)
print("Goal T:", T)
print("Goal q:", goal_q)
# Generate multiple diverse seeds (initial guesses)
# Sampled from typical ranges for UR5 joints
q_seeds = []
for shoulder in [0, np.pi]:
    for elbow in [-np.pi/2, np.pi/2]:
        for wrist in [-np.pi/2, np.pi/2]:
            q_seed = np.array([shoulder, -np.pi/4, elbow, 0, wrist, 0])
            q_seeds.append(q_seed)

# Use numerical IK to try each seed
solutions = []
seen = []

for q0 in q_seeds:
    sol = robot.ikine_LM(T, q0=q0)

    if sol.success:
        # Round to avoid numerical noise when checking for duplicates
        q_rounded = np.round(sol.q, decimals=4)
        if not any(np.allclose(q_rounded, s, atol=1e-3) for s in seen):
            solutions.append(sol.q)
            seen.append(q_rounded)

# Print the solutions
for i, q in enumerate(solutions):
    print(f"Solution {i+1}: {q}")
    # Print the forward kinematics for the solution
    T_sol = robot.fkine(q)
    print(f"Forward Kinematics for Solution {i+1}:\n {T_sol}")

sol = robot.ikine_LM(T, q0=goal_q)
print(f"Goal q: {sol.q}")
# Print solution transform
T_sol = robot.fkine(sol.q)
print(f"Forward Kinematics for Goal q:\n {T_sol}")

errors = [np.linalg.norm(q - goal_q) for q in solutions]
for i, err in enumerate(errors):
    print(f"Solution {i+1} error: {err:.6f}")
# Display solutions
# if len(solutions) == 0:
#     print("No valid IK solutions found.")
# else:
#     print(f"Found {len(solutions)} unique IK solutions.")
#     for i, q in enumerate(solutions):
#         print(f"\nSolution {i+1}: {q}")
#         # Print the forward kinematics for the solution
#         T_sol = robot.fkine(q)
#         print(f"Forward Kinematics for Solution {i+1}: {T_sol}")
#         robot.plot(q,  backend='pyplot', block=True)
#         # input("Press Enter to continue...")
