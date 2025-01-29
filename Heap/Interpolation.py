import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def symbolic_function(x, y, C = 1, a=20, b = 30):
    """
    Calculates the value of the constructed function.

    Args:
        x (float or np.ndarray): x coordinate(s).
        y (float or np.ndarray): y coordinate(s).
        C (float): scale coefficient.
        a (float): decay coefficient.
        b (float): width coefficient.

    Returns:
        float or np.ndarray: The function value(s).
    """
    term1 = (1-y)
    term2 = (1 - (x + 1)**2/4)
    term3 = (1 - (x - 1)**2/4)
    term4 = np.exp(-a * x**2)
    term5 = (1-np.exp(-b*(x-0.5)**2))
    term6 = (1-np.exp(-b*(x+0.5)**2))


    return C * term1 * term2 * term3 * term4 * term5 * term6

# Create grid of X and Y values
num_points = 100
x = np.linspace(-1, 1, num_points)
y = np.linspace(0, 1, num_points)
X, Y = np.meshgrid(x, y)

# Calculate Z values
Z = symbolic_function(X, Y)


# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))
#contour = ax.contourf(X, Y, Z, levels=20, cmap=cm.viridis)
ax = fig.add_subplot(111, projection='3d')
contour=ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False)
fig.colorbar(contour, shrink=0.5, aspect=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Symbolic Function')
plt.show()