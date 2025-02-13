import numpy as np
import matplotlib.pyplot as plt

# Define two ranges: one for positive values and one approaching zero
p_positive = np.linspace(0.001, 10, 1000)  # For regular plotting
p_near_zero = np.logspace(-10, -3, 1000)   # For showing asymptotic behavior
p = np.concatenate([p_near_zero, p_positive])

# Calculate function values
y = np.ma.masked_invalid(0.033 * np.log(p) + 0.865)
dy_dp = np.ma.masked_invalid(33 / (1000 * p))

# Create figure
plt.figure(figsize=(15, 6))

# Plot y = 0.033ln(p) + 0.865
plt.subplot(1, 2, 1)
plt.plot(p, y, label=r'$y = 0.033 \ln(p) + 0.865$', color='b')
plt.xlabel('p')
plt.ylabel('y')
plt.title('Function Plot')
plt.legend()
plt.grid()
plt.xlim(-1, 10)
plt.ylim(0.4, 1)  # Adjusted to show the relevant range
plt.xticks(np.arange(-1, 11, 2))
plt.yticks(np.arange(0.4, 1.1, 0.1))

# Plot derivative dy/dp
plt.subplot(1, 2, 2)
plt.plot(p, dy_dp, label=r'$\frac{dy}{dp} = \frac{33}{1000p}$', color='r')
plt.xlabel('p')
plt.ylabel('dy/dp')
plt.title('Derivative Plot')
plt.legend()
plt.grid()
plt.xlim(-1, 10)
plt.ylim(-0.2, 1)  # Updated y limits
plt.xticks(np.arange(-1, 11, 2))
plt.yticks(np.arange(-0.2, 1.2, 0.2))  # Updated y ticks

plt.tight_layout()
plt.show()
