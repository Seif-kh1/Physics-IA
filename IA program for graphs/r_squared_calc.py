import numpy as np
from data_analysis import df_cor_analysis, pressures_psi

# Get x and y values from the existing data
pressures = df_cor_analysis['Pressure (PSI)'].values
avg_cors = df_cor_analysis['Average COR'].values

# Get the logarithmic fit coefficients that were already calculated
z = np.polyfit(np.log(pressures), avg_cors, 1)

# Calculate predicted values using the existing fit
y_pred = z[0] * np.log(pressures) + z[1]

# Calculate R-squared
ss_res = np.sum((avg_cors - y_pred) ** 2)
ss_tot = np.sum((avg_cors - np.mean(avg_cors)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"Logarithmic equation: y = {z[0]:.6f}ln(x) + {z[1]:.6f}")
print(f"R² value: {r_squared:.6f}")
