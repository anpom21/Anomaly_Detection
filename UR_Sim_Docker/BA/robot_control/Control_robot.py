from __future__ import division, print_function
from dmp_position import PositionDMP
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import roboticstoolbox as rtb
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator


from spatialmath import SE3
from rtde_control import RTDEControlInterface
import time

def euclidean_distance(f1,f2):
    diff_x = f1[0] - f2[0]
    diff_y = f1[1] - f2[1]
    diff_z = f1[2] - f2[2]
    return np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)

def rmse(array):
    squared_diff = []
    for i in range(len(array)):
        squared_diff.append(array[i]  ** 2)
    mean_squared_diff = np.mean(squared_diff, axis=0)
    rmse = np.sqrt(mean_squared_diff)
    return rmse



if __name__ == '__main__':
    # Load file containing demonstration data
    demo = np.loadtxt("square_traj_90.dat", delimiter=" ")
    
    # Define time parameters
    dt=1/30 # 0.002
    tau = dt * len(demo)
    t = np.arange(0, tau, dt)
    # Extract position data from demonstration
    demo_p = demo[:, 0:3]
    
    
    #make copy of demo_p for plotting
    demo_orig=demo_p.copy()
    
    # Interpolation for demo_p
    cs_x=PchipInterpolator(t, demo_p)
    dt_2=1/500
    t_2=np.arange(0, tau, dt_2)
    demo_p=cs_x(t_2)
    
    #Chose parameters
    N = 50
    alph = 48
    dmp = PositionDMP(n_bfs=N, alpha=alph)
    dmp.train(demo_p, t_2, tau)

     
    #define ip of the robot
    #ip = "192.168.56.101" # For simulated machine
    ip = "192.168.0.10" # For real robot
    robot_control = RTDEControlInterface(ip)
 
    # Parameters for robot control
    velocity = 0.5
    acceleration = 0.5
    lookahead_time = 0.1
    gain = 300 
    axis_angle = [2.467, -1.767, 0.633]
    
    # Move to starting position
    input("Press enter to move to starting position")
    robot_control.moveL(demo_p[0].tolist() + axis_angle) 
    
    
    # Begin DMP movement
    input("Press enter to start DMP")
    
    
    dmp.reset()
    ps = []
    dps = []
    ddps = []
    tau=tau
    x = dmp.cs.step(dt_2, tau)
    while x > 1e-11:
        t_start = robot_control.initPeriod()
        x = dmp.cs.step(dt_2, tau)
        p, dp, ddp = dmp.step(x, dt_2, tau)
        
        ps.append(p.copy())
        dps.append(dp)
        ddps.append(ddp)
        pi = p.tolist() + axis_angle
        robot_control.servoL(pi, velocity, acceleration, dt_2, lookahead_time, gain)
        robot_control.waitPeriod(t_start)
    robot_control.servoStop()
    dmp_p = np.array(ps)
    
    
    # Plot the DMP against the original demonstration in 3D
    fig2 = plt.figure(2)
    print(demo_p[-1])
    print(dmp_p[-1])
    ax = plt.axes(projection='3d')
    ax.plot3D(demo_p[:, 0], demo_p[:, 1], demo_p[:, 2], label='Demonstration')
    ax.plot3D(dmp_p[:, 0], dmp_p[:, 1], dmp_p[:, 2], label='DMP')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.axis('equal')
    ax.legend()
    plt.show()
    
    

        