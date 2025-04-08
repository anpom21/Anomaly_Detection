from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

ip = "192.168.1.30"

rtde_c = RTDEControlInterface(ip)
rtde_r = RTDEReceiveInterface(ip)

print("Pose:", rtde_r.getActualTCPPose())
