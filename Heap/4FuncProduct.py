import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def f1(y, m):
    """
    Calculates the value of the function F1(y).
    """
    return ((-y + 1) / 2)**m * ((1 + y) / 2)

def f2(x, m):
    """
    Calculates the value of the function F2(x).
    """
    return ((x + 1) / 2)**m * ((1 - x) / 2)


def f3(x, m):
    """
    Calculates the value of the function F3(x).
    """
    return ((-x + 1) / 2)**m * ((1 + x) / 2)



def plot_composite_function_isolines(m, n, x_range=(-1, 1), y_range=(-1, 1)):
    """
    Plots isolines for the composite function Z(x,y)= F1(x) * F2(y) * F3(y).

    Args:
        m (int): The exponent value.
        x_range (tuple): A tuple for x values (min_x, max_x)
        y_range (tuple): A tuple for y values (min_y, max_y)

    Returns:
        None: Displays the plot.
    """
    # Create grid of X and Y values
    x = np.linspace(x_range[0], x_range[1], 200)
    y = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x, y)

    # Calculate the composite function Z
    Z = f1(Y, m) * f2(X, n) * f3(X, n) 

    # Normalize to have maximum 1
    max_z = np.max(Z)
    if max_z > 0:
       Z_normalized = Z / max_z
    else:
       Z_normalized = Z


    # Plot isolines
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contour(X, Y, Z_normalized, cmap=cm.viridis, levels=100)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Isolines of Z(x,y) = F1(x) * F2(y) * F3(y)')
    plt.show()


# Example usage
m = 3 # степени по y
n = 10 # степени по x
plot_composite_function_isolines(m,n)

