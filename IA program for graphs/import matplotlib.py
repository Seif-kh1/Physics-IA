import matplotlib.pyplot as plt
import numpy as np
from data_analysis import df_cor_analysis  # Import the existing DataFrame

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.18, 7.38))

# Plot 1: Original Average COR vs Pressure
pressures = df_cor_analysis['Pressure (PSI)'].values
avg_cors = df_cor_analysis['Average COR'].values
abs_uncertainties = df_cor_analysis['Avg Absolute Uncertainty'].values

# Add logarithmic fit
z = np.polyfit(np.log(pressures), avg_cors, 1)
equation = f'y = {z[0]:.3f}ln(x) + {z[1]:.3f}'

# Plot data with error bars
line = ax1.errorbar(pressures, avg_cors, yerr=abs_uncertainties,
                   fmt='o', label=f'Average COR\n{equation}',
                   markersize=5, capsize=3, capthick=1, elinewidth=1)

# Plot logarithmic fit
x_fit = np.linspace(min(pressures), max(pressures), 100)
ax1.plot(x_fit, z[0] * np.log(x_fit) + z[1], '--', alpha=0.5,
         color=line.lines[0].get_color())

ax1.set_xlabel('Internal Pressure (PSI)')
ax1.set_ylabel('Average Coefficient of Restitution (COR)')
ax1.set_title('Pressure vs. Average COR')
ax1.legend()
ax1.grid(True)

# Plot 2: Linearized form - modified to match graph 7's style
# Use the same x_fit range as graph 7
x_fit = np.linspace(min(pressures), max(pressures), 100)
y = 0.033 * x_fit + 0.865

ax2.plot(x_fit, y, '--', alpha=0.5, label='y = 0.033x + 0.865')
line = ax2.errorbar(pressures, avg_cors, yerr=abs_uncertainties,
                   fmt='o', label='Data points',
                   markersize=5, capsize=3, capthick=1, elinewidth=1)

# Match the axis limits and style of graph 7
ax2.set_xlabel('Internal Pressure (PSI)')
ax2.set_ylabel('Average Coefficient of Restitution (COR)')
ax2.set_title('Linearized Form of Pressure vs. Average COR')
ax2.legend()
ax2.grid(True)

# Set y-axis ticks for the linearized plot (ax2) from 0.86 to 0.94
ax2.yaxis.set_ticks(np.arange(0.85, 0.95, 0.02))
ax2.set_ylim(0.85, 0.95)

# Adjust layout and display
plt.tight_layout()
plt.show()
