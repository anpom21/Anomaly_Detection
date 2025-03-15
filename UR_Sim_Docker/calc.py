import numpy as np


x = -491
y = -133 
calc_ang = np.degrees(np.arctan2(y, x))
actual_ang = 0
print("X: ", x)
print("Y: ", y)
print(f"Calculated angle: {calc_ang} degrees")
print(f"Actual angle: {actual_ang} degrees")
print(f"Difference: {calc_ang - actual_ang}")

print("_____________________________")
x = 133
y = -491 
calc_ang = np.degrees(np.arctan2(y, x))
actual_ang = 90
print("X: ", x)
print("Y: ", y)
print(f"Calculated angle: {calc_ang} degrees")
print(f"Actual angle: {actual_ang} degrees")
print(f"Difference: {calc_ang - actual_ang}")

print("_____________________________")
x = 442
y = -253 
calc_ang = np.degrees(np.arctan2(y, x))
actual_ang = 135
print("X: ", x)
print("Y: ", y)
print(f"Calculated angle: {calc_ang} degrees")
print(f"Actual angle: {actual_ang} degrees")
print(f"Difference: {calc_ang - actual_ang}")
print("Radius: ", np.sqrt(x**2 + y**2))

print("_____________________________")

x = 300
y = 411 
calc_ang = np.degrees(np.arctan2(y, x))
actual_ang = 218
print("X: ", x)
print("Y: ", y)
print(f"Calculated angle: {calc_ang} degrees")
print(f"Actual angle: {actual_ang} degrees")
print(f"Difference: {calc_ang - actual_ang}")
print("Radius: ", np.sqrt(x**2 + y**2))

print("_____________________________")

x = -316
y = -399
calc_ang = np.degrees(np.arctan2(y, x))
actual_ang = -323
print("X: ", x)
print("Y: ", y)
print(f"Calculated angle: {calc_ang} degrees")
print(f"Actual angle: {actual_ang} degrees")
print(f"Difference: {calc_ang - actual_ang}")
print("Radius: ", np.sqrt(x**2 + y**2))
print("_____________________________")

x = 509
y = 0
calc_ang = np.degrees(np.arctan2(y, x))
actual_ang = -195
print("X: ", x)
print("Y: ", y)
print(f"Calculated angle: {calc_ang} degrees")
print(f"Actual angle: {actual_ang} degrees")
print(f"Difference: {calc_ang - actual_ang}")
print("_____________________________")

x = 0
y = 509
calc_ang = np.degrees(np.arctan2(y, x))
actual_ang = -105
print("X: ", x)
print("Y: ", y)
print(f"Calculated angle: {calc_ang} degrees")
print(f"Actual angle: {actual_ang} degrees")
print(f"Difference: {calc_ang - actual_ang}")
print("_____________________________")

def predict_angle(x, y):
    sol_1 = np.degrees(np.arctan2(y, x)) - 165
    sol_2 = np.degrees(np.arctan2(y, x)) + 195
    return np.degrees(np.arctan2(y, x))
