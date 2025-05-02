import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import matplotlib.pyplot as plt

# Load UR5 model
robot = rtb.models.UR5()

# Target end-effector pose
T = sm.SE3(0.5, -0.2, 0.3) * sm.SE3.Ry(np.pi)
print("T:", T)

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
