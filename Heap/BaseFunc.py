import numpy as np
import matplotlib.pyplot as plt

def n_m_modified(x, m):
    """
    Calculates the value of the modified function
     N_m_modified(x) =  ((x + 1) / 2)^m * ((1 - x) / 2)
    """
    return ((-x + 1) / 2)**m * ((1 + x) / 2)**m

def plot_n_m_functions_modified(m_range, a_m_range,x_range=(-1, 1)):
  """
  Plots the modified N_m functions for a range of m values.

  Args:
    m_range (list): a list of integer values for m.
    x_range (tuple): a tuple of (x_min, x_max) defining the range for x.
  """

  x = np.linspace(x_range[0], x_range[1], 400)
  plt.figure(figsize=(10, 6))
  plt.title('Plots of N_m(x) with Domain from -1 to 1')
  plt.xlabel('x')
  plt.ylabel('N_m(x)')

  for m in m_range:
      y = a_m_range[m-1]*n_m_modified(x, m)
      plt.plot(x, y, label=f'N_{m}(x)')

  plt.grid(True)
  plt.legend()
  plt.show()

# Example usage
m_range = [1, 2, 3, 4, 5, 6,7,8,9,10]
a_m_range=[-0.05,1,1,1,1,1,1,1,1,1]
plot_n_m_functions_modified(m_range, a_m_range)
