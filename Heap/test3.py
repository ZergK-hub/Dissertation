import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from matplotlib import cm


def create_boundary_data():
    """Creates a 5x5 grid of data values based on the given conditions."""
    x_nodes = np.linspace(-1, 1, 5)
    y_nodes = np.linspace(0, 1, 5)
    data_values = np.zeros((5, 5), dtype=float)

    for i, x in enumerate(x_nodes):
        for j, y in enumerate(y_nodes):
           if (y ==0 and -0.5 < x < 0.5):
               data_values[j,i] = 1
           else:
              data_values[j,i] = 0
    return x_nodes, y_nodes, data_values

def approximate_function(x,y, spline):
    """Evaluates a bicubic spline approximation at points x,y"""
    return spline(y, x, grid=False)


# 1. Create initial data and a spline function
x_nodes, y_nodes, data_values = create_boundary_data()
spline = RectBivariateSpline(y_nodes, x_nodes, data_values)


# 2. Evaluate a grid of data for visualisation
num_points = 100
x_new = np.linspace(-1, 1, num_points)
y_new = np.linspace(0, 1, num_points)
X, Y = np.meshgrid(x_new, y_new)

# 3. Use analytical function to plot
Z = approximate_function(X,Y, spline)

# 4. Plot result
fig, ax = plt.subplots(figsize=(8, 8))
contour = ax.contourf(X, Y, Z, levels=20, cmap=cm.viridis)
fig.colorbar(contour, shrink=0.5, aspect=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Bicubic Spline Approximation')
plt.show()