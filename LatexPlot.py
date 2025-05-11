import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

plt.plot([1, 2, 3], [4, 5, 6])
plt.title('Latex title')
plt.savefig('test_plot.png')
plt.show()