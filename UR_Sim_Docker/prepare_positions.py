# ---------------------------------------------------------------------------- #
#                                    IMPORT                                    #
# ---------------------------------------------------------------------------- #
from scipy.spatial.transform import Rotation
import numpy as np
from spatialmath import SE3
import spatialmath as sm
import matplotlib.pyplot as plt
from photometric_positions import half_sphere_simple, position_to_pose, plot_light_positions, plot_simple

# ---------------------------------------------------------------------------- #
#                                   FUNCTIONS                                  #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #
def main():
    print("Preparing positions...")
    # Parameters
    n_light = 4
    center = [0.379296019468532, -
              0.4164582402752736, -0.086414021169025]  # 0.3
    light_radius = 0.2
    light_height = 0.18

    # ---- Calculate light positions ---- #

    # # # Half Sphere # # #
    pos_half_sphere = half_sphere_simple(
        n_light, center, light_height, light_radius, 60)

    # --- Position to pose transformation ---- #
    T_light = [position_to_pose(pos, center) for pos in pos_half_sphere]

    # --- Plotting --- #
    plot_simple(T_light, center, light_radius, light_height, n_light)
    plot_light_positions(T_light, center, light_radius, light_height, n_light)


if __name__ == '__main__':
    main()
