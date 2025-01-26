import numpy as np
import matplotlib.pyplot as plt

def n_m(x, m):
    """
    Calculates the value of the function N_m(x) = x^m * (1 - x).

    Args:
        x (float or np.ndarray): The input value(s).
        m (int): The exponent.

    Returns:
        float or np.ndarray: The function value(s).
    """
    return (1-x)**m

def plot_n_m_functions(m_range, x_range=(0, 1)):
  """
  Plots N_m functions for a range of m values.
  Args:
    m_range (list): a list of integer values for m.
    x_range (tuple): a tuple of (x_min, x_max) defining the range for x.
  """

  # Create x values
  x = np.linspace(x_range[0], x_range[1], 400)

  # Setup plot
  plt.figure(figsize=(10, 6))
  plt.title('Plots of N_m(x) = x^m * (1 - x) for m from 1 to 6')
  plt.xlabel('x')
  plt.ylabel('N_m(x)')

  for m in m_range:
      y = n_m(x, m)
      plt.plot(x, y, label=f'N_{m}(x)') # Plot and label the function

  plt.grid(True)
  plt.legend()
  plt.show()

# Example usage
m_range = [1, 2, 3, 4, 5, 6]
plot_n_m_functions(m_range)
