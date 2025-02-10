import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Data for pressure and rebound heights
pressures = [35.85, 31.02, 24.13, 17.23, 10.34]
bounces = [0, 1, 2, 3, 4, 5, 6]

# Rebound heights for each pressure
heights = {
    35.85: [1.500, 1.268, 0.939, 0.728, 0.614, 0.529, 0.351],
    31.02: [1.500, 0.937, 0.641, 0.545, 0.371, 0.282, 0.228],
    24.13: [1.500, 0.906, 0.544, 0.343, 0.24, 0.165, 0.157],
    17.23: [1.500, 0.834, 0.553, 0.366, 0.268, 0.244, 0.221],
    10.34: [1.500, 0.459, 0.271, 0.177, 0.152, 0.151, 0.145],
}

# Define exponential function for fitting
def exp_func(x, a, b):
    return a * np.exp(-b * x)

# Plotting the data
plt.figure(figsize=(8, 6))
for pressure, height in heights.items():
    # Plot data points with error bars and store the line object
    line = plt.errorbar(bounces, height, yerr=0.02, fmt='o', markersize=3, capsize=2, ecolor='gray')
    
    color = line[0].get_color()
    
    # Fit exponential curve and plot with matching color
    x_fit = np.linspace(0, 6, 100)
    popt, _ = curve_fit(exp_func, bounces, height)
    y_fit = exp_func(x_fit, *popt)
    equation = f'f(x) = {popt[0]:.3f}e^(-{popt[1]:.3f}n)'
    plt.plot(x_fit, y_fit, '--', color=color, alpha=0.5, 
             label=f'Pressure = {pressure} kPa\n{equation}')

plt.title("Average Rebound Height vs Number of Bounces for Different Pressures")
plt.xlabel("Number of Bounces")
plt.ylabel("Average Rebound Height (±0.02 m)")
plt.legend(title="Pressure (kPa)")
plt.grid(True)
plt.tight_layout()
plt.show()
