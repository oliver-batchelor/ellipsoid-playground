import numpy as np
import matplotlib.pyplot as plt

def plot_falloff_functions(x_range=(-5, 5), k=2, num_points=1000):
    """
    Plot unnormalized Gaussian, generalized Gaussian, and raised cosine functions
    on the same chart for comparison, using a single sigma value
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    
    
    # Add raised cosine functions
    # Standard raised cosine
    mask = np.abs(x) <= np.pi
    y = np.zeros_like(x)
    y[mask] = 0.5 * (1 + np.cos(x[mask]) )**k
    plt.plot(x, y, label=f'Raised Cosine k={k}', linestyle=':')
    
    
    plt.title('Comparison of Falloff Functions')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    plot_falloff_functions()



