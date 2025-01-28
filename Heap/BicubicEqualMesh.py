import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def bicubic_interpolation_4x4(data_values, x_range=(-1, 1), y_range=(-1, 1), num_points=100):
    """
    Performs bicubic spline interpolation on a 4x4 grid.

    Args:
        data_values (np.ndarray): A 4x4 array of function values.
        x_range (tuple): A tuple (x_min, x_max).
        y_range (tuple): A tuple (y_min, y_max).
        num_points (int): Number of points along each axis for the interpolated data.

    Returns:
        tuple: A tuple containing (X, Y, Z), where:
        - X is a 2D array of x-coordinates.
        - Y is a 2D array of y-coordinates.
        - Z is a 2D array of interpolated z-values.
    """
    # Create grid points for known values
    x_nodes = np.linspace(x_range[0], x_range[1], 5)
    y_nodes = np.linspace(y_range[0], y_range[1], 5)


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
data_values = np.array([
    [0, 0, 0,0, 0],
    [0, 6, 7,6, 0],
    [0, 10, 11, 10, 0],
    [0, 6, 7, 6,0],
    [0, 0, 0, 0,0]
])

X, Y, Z = bicubic_interpolation_4x4(data_values)

# Plot the result
fig, ax = plt.subplots(figsize=(8, 8))

#contour = ax.contourf(X,Y,Z, levels = 20, cmap = cm.viridis)
ax = fig.add_subplot(111, projection='3d')
contour=ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False)
fig.colorbar(contour, shrink =0.5, aspect = 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Bicubic Spline Interpolation')
plt.show()
