import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def onpick(event):
    """
    Callback for pick events: changes the clicked point's color to green.
    """
    artist = event.artist
    # Only proceed if the clicked artist is our scatter:
    if artist is scat:
        # event.ind is a list of point indices that were clicked
        for i in event.ind:
            # Change color for that index to green
            colors[i] = 'green'
        # Update the scatter object with the new colors
        scat.set_color(colors)
        # Redraw the figure
        plt.draw()


# Generate some random 3D data
np.random.seed(5)
x = np.random.rand(10)
y = np.random.rand(10)
z = np.random.rand(10)

# Initially, all points are yellow
colors = ['yellow'] * len(x)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a 3D scatter plot with picking enabled
scat = ax.scatter(x, y, z, c=colors, s=100, picker=True)

# Connect our pick-event callback
fig.canvas.mpl_connect('pick_event', onpick)

plt.show()
