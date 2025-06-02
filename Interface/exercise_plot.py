import numpy as np
import matplotlib.pyplot as plt
import csv
import os


def load_data(file_path):
    time_list = []
    pressure_list = []
    position_list = []
    velocity_list = []

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            time_list.append(float(row[0]))
            pressure_list.append(float(row[1]))
            position_list.append(float(row[2]))
            velocity_list.append(float(row[3]))

    return np.array(time_list), np.array(pressure_list), np.array(position_list), np.array(velocity_list)


# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Define the path to the CSV file
    file_path = 'hard_difficulty.csv'

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        exit(1)

    # Load data from the CSV file
    time_list, pressure_list, position_list, velocity_list = load_data(
        file_path)

    # Plotting the data
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time_list, pressure_list, label='Pressure', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (Pa)')
    plt.title('Pressure over Time')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_list, position_list, label='Position', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Position over Time')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_list, velocity_list, label='Velocity', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity over Time')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()
