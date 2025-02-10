import matplotlib.pyplot as plt
import numpy as np

# Data: internal pressure (kPa) and the corresponding coefficients of restitution
pressure = [35.85, 31.02, 24.13, 17.23, 10.34]
coefficients = [
    [0.919, 0.861, 0.88, 0.918, 0.928, 0.815],
    [0.791, 0.827, 0.922, 0.825, 0.872, 0.898],
    [0.777, 0.775, 0.794, 0.838, 0.829, 0.974],
    [0.746, 0.814, 0.813, 0.855, 0.956, 0.95],
    [0.553, 0.768, 0.81, 0.924, 0.997, 0.983]
]

# Create extended x range for smooth polynomial curves
x_smooth = np.linspace(0, 40, 200)

# Plotting with polynomial fits
plt.figure(figsize=(10, 6))

# Loop through the bounces to plot scattered points and polynomial fits
for i in range(6):  # 6 bounces
    bounce_values = [coefficients[j][i] for j in range(5)]
    
    # Calculate polynomial coefficients (2nd degree)
    poly_coefs = np.polyfit(pressure, bounce_values, 2)
    poly_fit = np.poly1d(poly_coefs)
    
    # Plot scattered points and polynomial fit
    plt.scatter(pressure, bounce_values, label=f'Bounce {i + 1} data')
    plt.plot(x_smooth, poly_fit(x_smooth), '--', label=f'Bounce {i + 1} fit')

# Labels and title
plt.xlabel('Internal Pressure (kPa)')
plt.ylabel('Coefficient of Restitution')
plt.title('Coefficient of Restitution vs Internal Pressure with Polynomial Fits')
plt.legend(title='Bounce Number', bbox_to_anchor=(1.05, 1), loc='upper left')

# Displaying the plot
plt.grid(True)
plt.tight_layout()
plt.show()
