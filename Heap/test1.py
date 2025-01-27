import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def ellipse_function(x, y, h, k, a, b, A):
    """
    Calculates the value of the function that generates concentric ellipses.

    Args:
        x (float or np.ndarray): x coordinate(s).
        y (float or np.ndarray): y coordinate(s).
        h (float): x-coordinate of the center.
        k (float): y-coordinate of the center.
        a (float): semi-major axis.
        b (float): semi-minor axis.
        A (float): Scale factor

    Returns:
        float or np.ndarray: The function value(s).
    """
    return A*((x - h) / a)**2 + ((y - k) / (b))**2

def plot_concentric_ellipses_isolines(h, k, a, b, A, x_range=(-5, 5), y_range=(-5, 5)):
    """
    Plots the isolines of a function that generates concentric ellipses.

    Args:
       h (float): x-coordinate of the center.
       k (float): y-coordinate of the center.
       a (float): semi-major axis.
       b (float): semi-minor axis.
       A (float): Scale factor
       x_range (tuple): A tuple for x values (min_x, max_x).
       y_range (tuple): A tuple for y values (min_y, max_y).


    Returns:
        None: Displays the plot.
    """
    # Create grid of X and Y values
    x = np.linspace(x_range[0], x_range[1], 200)
    y = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x, y)

    # Calculate Z values
    Z = ellipse_function(X, Y, h, k, a, b, A)

    # Create contour plot
    fig, ax = plt.subplots(figsize=(8, 8))
    contour = ax.contour(X, Y, Z, cmap=cm.viridis, levels=20)
    ax.clabel(contour, inline=True, fontsize=8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Isolines of Concentric Ellipses')
    plt.show()

# Example usage
h = 0    # Center x-coordinate
k = 0    # Center y-coordinate
a = 2    # Semi-major axis
b = 1    # Semi-minor axis
A = 30    # scale factor

plot_concentric_ellipses_isolines(h, k, a, b, A)
