import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def bicubic_interpolation_bias_mesh(data_values, x_nodes, y_nodes, x_range, y_range, num_points):
    """
    Performs bicubic spline interpolation on a bias mesh.

    Args:
        data_values (np.ndarray): A 2D array of function values.
        x_nodes (np.ndarray): 1D array of x-coordinates for known values.
        y_nodes (np.ndarray): 1D array of y-coordinates for known values.
        x_range (tuple): A tuple (x_min, x_max) for the interpolation range.
        y_range (tuple): A tuple (y_min, y_max) for the interpolation range.
        num_points (int): Number of points along each axis for the interpolated data.

    Returns:
        tuple: A tuple containing (X, Y, Z), where:
        - X is a 2D array of x-coordinates.
        - Y is a 2D array of y-coordinates.
        - Z is a 2D array of interpolated z-values.
    """
    # Create the interpolation function
    interp_func = interp2d(x_nodes, y_nodes, data_values, kind='cubic')

    # Create new x and y values
    x_new = np.linspace(x_range[0], x_range[1], num_points)
    y_new = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x_new, y_new)

    # Interpolate z values
    Z = interp_func(x_new, y_new)

    return X, Y, Z


# Example usage

# Example of a bias mesh:
x_nodes = np.array([-1, -0.5, 0.2, 1])
y_nodes = np.array([-1, -0.3, 0.6, 1])

data_values = np.array([
    [0, 0, 0, 0],
    [0, 6, 7, 0],
    [0, 10, 11, 0],
    [0, 0, 0, 0]
])


x_range = (-1.5, 1.5)
y_range = (-1.5, 1.5)
num_points= 100
X, Y, Z = bicubic_interpolation_bias_mesh(data_values, x_nodes, y_nodes, x_range, y_range, num_points)

# Plot the result
fig, ax = plt.subplots(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
contour=ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False)
#contour = ax.contourf(X,Y,Z, levels = 20, cmap = cm.viridis)
fig.colorbar(contour, shrink =0.5, aspect = 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Bicubic Spline Interpolation (Bias Mesh)')
plt.show()