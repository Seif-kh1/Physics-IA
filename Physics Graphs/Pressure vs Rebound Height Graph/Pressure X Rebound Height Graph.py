import numpy as np
import matplotlib.pyplot as plt

# Configure the figure
plt.figure(figsize=(12, 8), frameon=False)

pressure = np.array([35.85, 31.02, 24.13, 17.23, 10.34])
rebound_heights = np.array([
    [1.268, 0.939, 0.728, 0.614, 0.529, 0.351],
    [0.937, 0.641, 0.545, 0.371, 0.282, 0.228],
    [0.906, 0.544, 0.343, 0.24, 0.165, 0.157],
    [0.834, 0.553, 0.366, 0.268, 0.244, 0.221],
    [0.459, 0.271, 0.177, 0.152, 0.151, 0.145]
])

# List to store legend entries
legend_entries = []

# Create the plot with error bars and best fit lines
for i in range(len(rebound_heights[0])):
    # Plot points with error bars and store the plot object
    err_plot = plt.errorbar(pressure, rebound_heights[:, i], 
                           xerr=0.6,
                           yerr=0.02, 
                           fmt='o',
                           capsize=2,
                           markersize=4,
                           ecolor='gray')  # Added gray color for error bars
    
    # Calculate and plot best fit line with matching color
    z = np.polyfit(pressure, rebound_heights[:, i], 1)
    p = np.poly1d(z)
    line = plt.plot(pressure, p(pressure), '--', alpha=0.7, 
                   color=err_plot[0].get_color())[0]
    
    # Format equation string with bounce number
    equation = f'Bounce {i+1}: y = {z[0]:.3f}x + {z[1]:.3f}'
    legend_entries.append(equation)

# Add legend with adjusted position and title
plt.legend(legend_entries, loc='upper left', title='Best Fit Lines', 
          bbox_to_anchor=(0.02, 0.98))

plt.xlabel('Pressure (±0.6 kPa)')
plt.ylabel('Rebound Height (±0.02 m)')
plt.title('Pressure vs Rebound Height')
plt.grid(True)

# Set y-axis tick spacing to 0.1
plt.yticks(np.arange(0, plt.ylim()[1], 0.1))

# Adjust the layout to remove padding
plt.tight_layout()
# Turn off toolbar and window decorations
plt.rcParams['toolbar'] = 'None'
manager = plt.get_current_fig_manager()
manager.set_window_title("")  # Remove window title
try:
    manager.window.state('zoomed')  # For Windows
except:
    manager.full_screen_toggle()  # For other platforms

plt.show()
