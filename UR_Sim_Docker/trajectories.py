import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from rtde_control import RTDEControlInterface
import time

def move_to_start_configuration(robot_control, Q0, P0, axis_angle):
    input("Press enter to move to start configuration.")
    robot_control.moveJ(Q0)
    robot_control.moveL(P0.t.tolist() + axis_angle)

if __name__ == "__main__":
    # Initialize robot
    #ip = "172.17.0.2"  # TODO: Change this
    ip = "192.168.1.30"
    robot_control = RTDEControlInterface(ip)
    # robot_receive = RTDEReceiveInterface(ip)
    
    # Get current joint configuration
    # # Get current joint positions
    # current_joint_positions = robot_control.getActualToolFlangePose()
    # print("Current joint positions:", current_joint_positions)
    
    # # Get current Cartesian configuration
    # T = robot_control.getActualTCPPose()

    # Parameters
    velocity = 0.5
    acceleration = 0.5
    lookahead_time = 0.1
    gain = 300

    # Points
    P0 = SE3(0.36177, 0.10525, 0.64767)
    P1 = SE3(0.36177, -0.22975, 0.3162)
    P2 = SE3(0.36287, -0.42882, 0.53658)

    delta2 = P2.t - P1.t

    axis_angle = [2.467, -1.767, 0.633]

    # Joint configurations
    Q0 = np.array(
        [
            0.6743066906929016,
            -1.3727410475360315,
            -1.204346005116598,
            4.2224321365356445,
            1.159425139427185,
            0.3307466506958008,
        ]
    )
    Q1 = np.array(
        [
            -0.2682741324054163,
            -1.4053967634784144,
            -2.133138958607809,
            5.484482765197754,
            1.3903940916061401,
            -0.6350424925433558,
        ]
    )
    Q2 = np.array(
        [
            -0.6654709021197718,
            -1.6713898817645472,
            -1.312082592641012,
            4.966042518615723,
            1.5440434217453003,
            -1.002007786427633,
        ]
    )

    print("Q0: ", Q0)
    print("Q1: ", Q1)

    deltaQ2 = Q2 - Q1

    # Time parameters
    # dt = 0.002
    dt = 0.008
    T = 2
    t = np.arange(0, T, dt)

    # Cartesian space trapezoidal velocity
    T_trap_1 = rtb.tools.trajectory.ctraj(P0, P1, t)
    T_trap_2 = rtb.tools.trajectory.ctraj(P1, P2, t)

    move_to_start_configuration(robot_control, Q0, P0, axis_angle)
    
    

    input("Press enter to start Cartesian-space trapezoidal velocity profile")
    for i in range(0, len(T_trap_1)):
        pi = T_trap_1.t[i].tolist() + axis_angle
        robot_control.servoL(pi, velocity, acceleration, dt, lookahead_time, gain)
        time.sleep(dt)

    for i in range(0, len(T_trap_2)):
        pi = T_trap_2.t[i].tolist() + axis_angle
        robot_control.servoL(pi, velocity, acceleration, dt, lookahead_time, gain)
        time.sleep(dt)
    robot_control.servoStop()

    # Cartesian space 5th order polynomial
    T_poly_x_1 = rtb.tools.trajectory.quintic(P0.t[0], P1.t[0], t, qdf=delta2[0])
    T_poly_y_1 = rtb.tools.trajectory.quintic(P0.t[1], P1.t[1], t, qdf=delta2[1])
    T_poly_z_1 = rtb.tools.trajectory.quintic(P0.t[2], P1.t[2], t, qdf=delta2[2])

    T_poly_x_2 = rtb.tools.trajectory.quintic(P1.t[0], P2.t[0], t, qd0=delta2[0])
    T_poly_y_2 = rtb.tools.trajectory.quintic(P1.t[1], P2.t[1], t, qd0=delta2[1])
    T_poly_z_2 = rtb.tools.trajectory.quintic(P1.t[2], P2.t[2], t, qd0=delta2[2])

    input("Press enter to move to start configuration.")
    robot_control.moveJ(Q0)
    robot_control.moveL(P0.t.tolist() + axis_angle)

    print(T_poly_x_1.s)

    input("Press enter to start Cartesian-space 5th order polynomial")
    for i in range(0, len(T_poly_x_1)):
        pi = [T_poly_x_1.s[i], T_poly_y_1.s[i], T_poly_z_1.s[i]] + axis_angle
        robot_control.servoL(pi, velocity, acceleration, dt, lookahead_time, gain)
        time.sleep(dt)

    for i in range(0, len(T_poly_x_2)):
        pi = [T_poly_x_2.s[i], T_poly_y_2.s[i], T_poly_z_2.s[i]] + axis_angle
        robot_control.servoL(pi, velocity, acceleration, dt, lookahead_time, gain)
        time.sleep(dt)

    robot_control.servoStop()

    # Joint space 5th order polynomial
    Qs_1 = rtb.tools.trajectory.jtraj(Q0, Q1, len(t), qd1=deltaQ2)
    qs_1 = Qs_1.q

    Qs_2 = rtb.tools.trajectory.jtraj(Q1, Q2, len(t), qd0=deltaQ2)
    qs_2 = Qs_2.q

    input("Press enter to move to start configuration.")
    robot_control.moveJ(Q0)

    input("Press enter to start joint-space 5th order polynomial")
    for i in range(0, len(qs_1)):
        qi = qs_1[i]
        robot_control.servoJ(
            qi.tolist(), velocity, acceleration, dt, lookahead_time, gain
        )
        time.sleep(dt)

    for i in range(0, len(qs_2)):
        qi = qs_2[i]
        robot_control.servoJ(
            qi.tolist(), velocity, acceleration, dt, lookahead_time, gain
        )
        time.sleep(dt)
    robot_control.servoStop()
