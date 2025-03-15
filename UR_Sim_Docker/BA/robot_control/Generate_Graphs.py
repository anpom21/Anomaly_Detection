from __future__ import division, print_function
from dmp_position import PositionDMP
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator

# Calculate the Euclidean distance between two points
def euclidean_distance(f1,f2):
    diff_x = f1[0] - f2[0]
    diff_y = f1[1] - f2[1]
    diff_z = f1[2] - f2[2]
    return np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)

# Calculate the root mean square error of an array
def rmse(array):
    squared_diff = []
    for i in range(len(array)):
        squared_diff.append(array[i]  ** 2)
    mean_squared_diff = np.mean(squared_diff, axis=0)
    rmse = np.sqrt(mean_squared_diff)
    return rmse

# Calculate the root mean square error of a DMP compared to a trajectory
def rmse_dmp_demo(dmp_p, demo_p):
    euclidean_dist = []
    error=0
    for i in range(0,len(dmp_p)):
      euclidean_dist.append(euclidean_distance(demo_p[i], dmp_p[i]))
    return rmse(euclidean_dist),rmse(euclidean_dist[3000:len(euclidean_dist)-1])

# Calculate the root mean square error of a DMP compared to a demonstration
def calc_rmse(right, msg=""):
    wrist = right.copy()
    height = 0.344
    length = 0.431
    start_point = wrist[0]
    start_top = wrist[0] + [0, 0, height]
    end_point = wrist[0] + [0, -length, 0]
    # RMSE
    error = []
    for point in wrist:
        point_st = [start_point[0], start_point[1], point[2]]
        start_dist = np.linalg.norm(point_st - point)

        point_mid = [point[0], point[1], start_top[2]]
        mid_dist = np.linalg.norm(point_mid - point)

        point_end = [end_point[0], end_point[1], point[2]]
        end_dist = np.linalg.norm(point_end - point)

        # Append the smallest error
        error.append(min(start_dist, mid_dist, end_dist))
    # Calculate the RMSE of the error
    error = np.array(error)
    rmse = np.sqrt(np.mean(error**2))
    print(msg+f" RMSE: {rmse}")
    return rmse
    # RMSE

if __name__ == '__main__':
    # Load a demonstration file containing robot positions.
    demo = np.loadtxt("demo.dat", delimiter=" ")
    dt=1/30 # 0.002
    
    # Choose only the position data from the demonstration
    demo_p=demo[:,0:3]
    # Calculate the time constant tau from the demonstration and the time vector
    tau = dt * len(demo_p)
    t = np.arange(0, tau, dt)   
    # Interpolate the demonstration data to 500Hz
    cs_x=PchipInterpolator(t, demo_p)
    dt_2=1/500
    t_2=np.arange(0, tau, dt_2)
    interp=cs_x(t_2)
    
    # calculate the RMSE of the demonstration and the interpolated data compared to the original demonstration
    calc_rmse(demo_p, "Demonstration")
    calc_rmse(interp, "Interpolation")
    
    
    # Plot the demonstration and the interpolated data
    fig2 = plt.figure(2)
    ax = plt.axes(projection='3d')
    ax.plot3D(demo_p[:, 0], demo_p[:, 1], demo_p[:, 2], label='Demonstration with 30Hz')
    ax.plot3D(interp[:, 0], interp[:, 1], interp[:, 2], label='Interpolation with 500Hz')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    
    fig1, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(t, demo_p[:, 0], label='Demonstration with 30Hz')
    axs[0].plot(t_2, interp[:, 0], label='Interpolation with 500Hz')
    if t_2[-1] < t[-1]:
        axs[0].set_xlim([t[0],t[-1]])
    else:
        axs[0].set_xlim([t_2[0],t_2[-1]])
    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('X (m)')
    

    axs[1].plot(t, demo_p[:, 1], label='Demonstration with 30Hz')
    axs[1].plot(t_2, interp[:,1], label='Interpolation with 500Hz')
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('Y (m)')

    axs[2].plot(t, demo_p[:, 2], label='Demonstration with 30Hz')
    axs[2].plot(t_2, interp[:, 2], label='Interpolation with 500Hz')
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('Z (m)')
    axs[2].legend()
    plt.show()
    
    
    # Train a DMP on the interpolated data with chosen parameters for the number of basis functions and 
    N_1 = 20 
    N_2 = 35
    N_3 = 50
    N_4 = 65
    alph = 48 # beta=alpha/4
    
    dmp_1 = PositionDMP(n_bfs=N_1, alpha=alph)
    dmp_1.train(interp, t_2, tau)
    dmp_2 = PositionDMP(n_bfs=N_2, alpha=alph)
    dmp_2.train(interp, t_2, tau)
    dmp_3 = PositionDMP(n_bfs=N_3, alpha=alph)
    dmp_3.train(interp, t_2, tau)
    dmp_4 = PositionDMP(n_bfs=N_4, alpha=alph)
    dmp_4.train(interp, t_2, tau)
    
    # Set the start and goal points of the DMP
    dmp_1.p0=interp[0]
    dmp_2.p0=interp[0]-[0.1,0.1,0.0]
    dmp_3.p0=interp[0]+[0.1,0.1,0.0]
    dmp_4.p0=interp[0]+[0.3,0.3,0.0]
    
    dmp_1.gp=interp[-1]
    dmp_2.gp=interp[-1]-[0.1,0.1,0.0]
    dmp_3.gp=interp[-1]+[0.1,0.1,0.0]
    dmp_4.gp=interp[-1]+[0.3,0.3,0.0]
    
     

    # Choose the time constant tau for the rollout and generate time vector
    tau_roll=tau
    t_roll=np.arange(0, tau_roll, dt_2)
    
    # Generate the output trajectory from the trained DMP
    dmp_1_p, dmp_1_dp, dmp_1_ddp = dmp_1.rollout(t_roll, tau_roll)
    dmp_2_p, dmp_2_dp, dmp_2_ddp = dmp_2.rollout(t_roll, tau_roll)
    dmp_3_p, dmp_3_dp, dmp_3_ddp = dmp_3.rollout(t_roll, tau_roll)
    dmp_4_p, dmp_4_dp, dmp_4_ddp = dmp_4.rollout(t_roll, tau_roll)
    
    # Calculate the RMSE of each DMP compared to the demonstration
    error_1 , error_1_Corrected=rmse_dmp_demo(dmp_1_p, interp)
    error_2, error_2_Corrected=rmse_dmp_demo(dmp_2_p, interp)
    error_3, error_3_Corrected=rmse_dmp_demo(dmp_3_p, interp)
    error_4, error_4_Corrected=rmse_dmp_demo(dmp_4_p, interp)
    
    #round the errors
    print("errors")
    error_1=round(error_1, 5)
    error_2=round(error_2, 5)
    error_3=round(error_3, 5)
    error_4=round(error_4, 5)
    error_1_Corrected=round(error_1_Corrected, 5)
    error_2_Corrected=round(error_2_Corrected, 5)
    error_3_Corrected=round(error_3_Corrected, 5)
    error_4_Corrected=round(error_4_Corrected, 5)
    
    # Print the euclidean distance between the end point of the demonstration and the end point of each DMP
    print("error in x-y-z")
    print("n=20")
    print(dmp_1.gp-dmp_1_p[-1], euclidean_distance(dmp_1.gp, dmp_1_p[-1]))   
    print("n=35")
    print(dmp_2.gp-dmp_2_p[-1], euclidean_distance(dmp_2.gp, dmp_2_p[-1]))
    print("n=50")
    print(dmp_3.gp-dmp_3_p[-1], euclidean_distance(dmp_3.gp, dmp_3_p[-1]))
    print("n=65")
    print(dmp_4.gp-dmp_4_p[-1],euclidean_distance(dmp_4.gp, dmp_4_p[-1]))
    
    # Plot the demonstration and the DMP output with varying parameters as chosen above
    fig1, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(t, demo_p[:, 0], label='Demonstration')
    axs[0].plot(t_roll, dmp_1_p[:, 0], label='DMP 1')
    #axs[0].plot(t*9, dmp_p[:, 0], label='DMP fitted')
    if t_roll[-1] < t[-1]:
        axs[0].set_xlim([t[0],t[-1]])
    else:
        axs[0].set_xlim([t_roll[0],t_roll[-1]])
    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('X (m)')
    

    axs[1].plot(t_roll, interp[:, 1], label='Demonstration')
    axs[1].plot(t_roll, dmp_1_p[:, 1], label='DMP 1')
    #axs[1].plot(t*time_extention, dmp_p[:, 1], label='DMP fitted')
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('Y (m)')

    axs[2].plot(t, demo_p[:, 2], label='Demonstration')
    axs[2].plot(t_roll, dmp_1_p[:, 2], label='DMP 1')
  #  axs[2].plot(t*time_extention, dmp_p[:, 2], label='DMP fitted')
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('Z (m)')
    plt.plot([], [], ' ', label=f'RMSE: {error_1}')
    plt.plot([], [], ' ', label=f'RMSE after 6s: {error_1_Corrected}')
    axs[2].legend()

    fig3, axs3 = plt.subplots(3, 1, sharex=True)
    axs3[0].plot(t, demo_p[:, 0], label='Demonstration')
    axs3[0].plot(t_roll, dmp_2_p[:, 0], label='DMP 2')
    #axs[0].plot(t*9, dmp_p[:, 0], label='DMP fitted')
    if t_roll[-1] < t[-1]:
        axs3[0].set_xlim([t[0],t[-1]])
    else:
        axs3[0].set_xlim([t_roll[0],t_roll[-1]])
    axs3[0].set_xlabel('t (s)')
    axs3[0].set_ylabel('X (m)')

    axs3[1].plot(t_roll, interp[:, 1], label='Demonstration')
    axs3[1].plot(t_roll, dmp_2_p[:, 1], label='DMp 2')
    #axs[1].plot(t*time_extention, dmp_p[:, 1], label='DMP fitted')
    axs3[1].set_xlabel('t (s)')
    axs3[1].set_ylabel('Y (m)')

    axs3[2].plot(t, demo_p[:, 2], label='Demonstration')
    axs3[2].plot(t_roll, dmp_2_p[:, 2], label='DMP 2')
  #  axs[2].plot(t*time_extention, dmp_p[:, 2], label='DMP fitted')
    axs3[2].set_xlabel('t (s)')
    axs3[2].set_ylabel('Z (m)')
    plt.plot([], [], ' ', label=f'RMSE: {error_2}')
    plt.plot([], [], ' ', label=f'RMSE after 6s: {error_2_Corrected}')
    
    axs3[2].legend()
    
    fig4, axs4 = plt.subplots(3, 1, sharex=True)
    axs4[0].plot(t, demo_p[:, 0], label='Demonstration')
    axs4[0].plot(t_roll, dmp_3_p[:, 0], label='DMP 3')
    #axs[0].plot(t*9, dmp_p[:, 0], label='DMP fitted')
    if t_roll[-1] < t[-1]:
        axs4[0].set_xlim([t[0],t[-1]])
    else:
        axs4[0].set_xlim([t_roll[0],t_roll[-1]])
    axs4[0].set_xlabel('t (s)')
    axs4[0].set_ylabel('X (m)')

    axs4[1].plot(t_roll, interp[:, 1], label='Demonstration')
    axs4[1].plot(t_roll, dmp_3_p[:, 1], label='DMp 3')
    #axs[1].plot(t*time_extention, dmp_p[:, 1], label='DMP fitted')
    axs4[1].set_xlabel('t (s)')
    axs4[1].set_ylabel('Y (m)')

    axs4[2].plot(t, demo_p[:, 2], label='Demonstration')
    axs4[2].plot(t_roll, dmp_3_p[:, 2], label='DMP 3')
  #  axs[2].plot(t*time_extention, dmp_p[:, 2], label='DMP fitted')
    axs4[2].set_xlabel('t (s)')
    axs4[2].set_ylabel('Z (m)')
    plt.plot([], [], ' ', label=f'RMSE: {error_3}')
    plt.plot([], [], ' ', label=f'RMSE after 6s: {error_3_Corrected}')

    axs4[2].legend()
    
    fig5, axs5 = plt.subplots(3, 1, sharex=True)
    axs5[0].plot(t, demo_p[:, 0], label='Demonstration')
    axs5[0].plot(t_roll, dmp_4_p[:, 0], label='DMP 4')
    #axs[0].plot(t*9, dmp_p[:, 0], label='DMP fitted')
    if t_roll[-1] < t[-1]:
        axs5[0].set_xlim([t[0],t[-1]])
    else:
        axs5[0].set_xlim([t_roll[0],t_roll[-1]])
    axs5[0].set_xlabel('t (s)')
    axs5[0].set_ylabel('X (m)')

    axs5[1].plot(t_roll, interp[:, 1], label='Demonstration')
    axs5[1].plot(t_roll, dmp_4_p[:, 1], label='DMp 4')
    #axs[1].plot(t*time_extention, dmp_p[:, 1], label='DMP fitted')
    axs5[1].set_xlabel('t (s)')
    axs5[1].set_ylabel('Y (m)')

    axs5[2].plot(t, demo_p[:, 2], label='Demonstration')
    axs5[2].plot(t_roll, dmp_4_p[:, 2], label='DMP 4')
  #  axs[2].plot(t*time_extention, dmp_p[:, 2], label='DMP fitted')
    axs5[2].set_xlabel('t (s)')
    axs5[2].set_ylabel('Z (m)')
    plt.plot([], [], ' ', label=f'RMSE: {error_4}')
    plt.plot([], [], ' ', label=f'RMSE after 6s: {error_4_Corrected}')
    axs5[2].legend()
    plt.show()
    
        
    
    # Generate the output trajectory from the trained DMP 2 with different time constants
    tau_roll_05=tau*0.5
    tau_roll_2=tau*2
    
    
    t_roll_05=np.arange(0, tau_roll_05, dt_2)
    t_roll_2=np.arange(0, tau_roll_2, dt_2)
    
    dmp_p_05, dmp_dp_05, dmp_ddp_05 = dmp_2.rollout(t_roll_05, tau_roll_05)
    dmp_p_2, dmp_dp_2, dmp_ddp_2 = dmp_2.rollout(t_roll_2, tau_roll_2)
    
    # Calculate the euclidean distances of the DMP with different time constants compared to the demonstration
    
    
    print("error in x-y-z")
    print("tau * 0.5")
    print(interp[-1]-dmp_p_05[-1], euclidean_distance(interp[-1], dmp_p_05[-1]))
    print("tau / 2")
    print(interp[-1]-dmp_p_2[-1], euclidean_distance(interp[-1], dmp_p_2[-1]))
    print("original tau")
    print(interp[-1]-dmp_2_p[-1],euclidean_distance(interp[-1], dmp_2_p[-1]))
    
    # Plot the demonstration and the DMP output with different time constants
    fig3, axs3 = plt.subplots(3, 1, sharex=True)
    axs3[0].plot(t, demo_p[:, 0], label='Demonstration')
    axs3[0].plot(t_roll, dmp_2_p[:, 0], label='DMP with normal tau')
    axs3[0].plot(t_roll_05, dmp_p_05[:, 0], label='DMP with tau*0.5')
    axs3[0].plot(t_roll_2, dmp_p_2[:, 0], label='DMP with tau*2')
    #axs[0].plot(t*9, dmp_p[:, 0], label='DMP fitted')
    if t_roll[-1] < t[-1]:
        axs3[0].set_xlim([t[0],t[-1]])
    else:
        axs3[0].set_xlim([t_roll[0],t_roll[-1]])
    axs3[0].set_xlabel('t (s)')
    axs3[0].set_ylabel('X (m)')

    axs3[1].plot(t_roll, interp[:, 1], label='Demonstration')
    axs3[1].plot(t_roll, dmp_2_p[:, 1], label='DMP with normal tau')
    axs3[1].plot(t_roll_05, dmp_p_05[:, 1], label='DMP with tau*0.5')
    axs3[1].plot(t_roll_2, dmp_p_2[:, 1], label='DMP with tau*2')
    axs3[1].set_xlabel('t (s)')
    axs3[1].set_ylabel('Y (m)')

    axs3[2].plot(t, demo_p[:, 2], label='Demonstration')
    axs3[2].plot(t_roll, dmp_2_p[:, 2], label='DMP with normal tau')
    axs3[2].plot(t_roll_05, dmp_p_05[:, 2], label='DMP with tau*0.5')
    axs3[2].plot(t_roll_2, dmp_p_2[:, 2], label='DMP with tau*2')
    axs3[2].set_xlabel('t (s)')
    axs3[2].set_ylabel('Z (m)')
    axs3[2].legend()
    plt.show()
    
    
    # Plot the demonstration, the reference and the DMP output with different start positions in 3D
    fig2 = plt.figure(2)
    ax = plt.axes(projection='3d')

    ax.plot3D(dmp_1_p[:, 0], dmp_1_p[:, 1], dmp_1_p[:, 2], label='DMP with original start')
    ax.plot3D(dmp_2_p[:, 0], dmp_2_p[:, 1], dmp_2_p[:, 2], label='DMP with start - [0.1,0.1,0.0]')
    ax.plot3D(dmp_3_p[:, 0], dmp_3_p[:, 1], dmp_3_p[:, 2], label='DMP with start + [0.1,0.1,0.0]')
    ax.plot3D(dmp_4_p[:, 0], dmp_4_p[:, 1], dmp_4_p[:, 2], label='DMP with start + [0.3,0.3,0.0]')

    # Generate the reference trajectory
    ax.plot3D(demo_p[:, 0], demo_p[:, 1], demo_p[:, 2], label='Demonstration')
    height = 0.344
    length = 0.431

    start_point = demo_p[0]
    start_top = demo_p[0] + [0, 0, height]
    end_top = demo_p[0] + [0, -length, height]
    end_point = demo_p[0] + [0, -length, 0]
    reference = np.array([start_point, start_top, end_top, end_point])
    ax.plot3D(reference[:, 0], reference[:, 1], reference[:, 2], label='Reference')
    ax.view_init(elev=22, azim=-43)
    plt.axis('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(loc='upper right')
    plt.show()
    
    # Calculate the RMSE of each DMP compared to the reference
    calc_rmse(dmp_1_p, "DMP with 20 basis functions")
    calc_rmse(dmp_2_p, "DMP with 35 basis functions")
    calc_rmse(dmp_3_p, "DMP with 50 basis functions")
    calc_rmse(dmp_4_p, "DMP with 65 basis functions")
    
    # Calculate the RMSE of each DMP compared to the reference after 6 seconds
    calc_rmse(dmp_1_p[3000:len(dmp_1_p)-1,:], "DMP with 20 basis functions")
    calc_rmse(dmp_2_p[3000:len(dmp_2_p)-1,:], "DMP with 35 basis functions")
    calc_rmse(dmp_3_p[3000:len(dmp_3_p)-1,:], "DMP with 50 basis functions")
    calc_rmse(dmp_4_p[3000:len(dmp_4_p)-1,:], "DMP with 65 basis functions")