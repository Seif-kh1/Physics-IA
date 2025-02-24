import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create folders for graphs and Excel files
graphs_folder = os.path.join(script_dir, 'graphs')
excel_folder = os.path.join(script_dir, 'excel_files')

# Create directories if they don't exist
os.makedirs(graphs_folder, exist_ok=True)
os.makedirs(excel_folder, exist_ok=True)

# COR calculation function
def calculate_cor(heights):
    # Convert list to numpy array before calculation
    heights_array = np.array(heights)
    return np.sqrt(heights_array[1:] / heights_array[:-1])

# Given data
pressures_psi = [6.5, 5.5, 4.5, 3.5, 2.5, 1.5]
pressures_kpa = [44.81, 37.92, 31.02, 24.13, 17.23, 10.34]
bounces = np.arange(1, 7)
h0 = 1.7  # Initial drop height in meters

# Rebound heights for each pressure (averaged over 5 trials)
rebound_heights = {
    6.5: np.mean([[1.52, 1.32, 1.14, 0.98, 0.83, 0.7],
                  [1.5, 1.3, 1.12, 0.96, 0.82, 0.68],
                  [1.49, 1.29, 1.11, 0.95, 0.81, 0.67],
                  [1.53, 1.33, 1.15, 0.99, 0.84, 0.71],
                  [1.51, 1.31, 1.13, 0.97, 0.83, 0.69]], axis=0),
    5.5: np.mean([[1.43, 1.24, 1.07, 0.92, 0.78, 0.65],
                  [1.41, 1.22, 1.05, 0.9, 0.76, 0.63],
                  [1.42, 1.23, 1.06, 0.91, 0.77, 0.64],
                  [1.44, 1.25, 1.08, 0.93, 0.79, 0.66],
                  [1.42, 1.23, 1.06, 0.91, 0.77, 0.64]], axis=0),
    4.5: np.mean([[1.32, 1.14, 0.98, 0.84, 0.71, 0.59],
                  [1.3, 1.12, 0.96, 0.82, 0.69, 0.57],
                  [1.31, 1.13, 0.97, 0.83, 0.7, 0.58],
                  [1.33, 1.15, 0.99, 0.85, 0.72, 0.6],
                  [1.31, 1.13, 0.97, 0.83, 0.7, 0.58]], axis=0),
    3.5: np.mean([[1.18, 1.02, 0.87, 0.74, 0.62, 0.51],
                  [1.16, 1, 0.85, 0.72, 0.6, 0.49],
                  [1.17, 1.01, 0.86, 0.73, 0.61, 0.5],
                  [1.19, 1.03, 0.88, 0.75, 0.63, 0.52],
                  [1.17, 1.01, 0.86, 0.73, 0.61, 0.5]], axis=0),
    2.5: np.mean([[1, 0.86, 0.73, 0.62, 0.52, 0.43],
                  [0.98, 0.84, 0.71, 0.6, 0.5, 0.41],
                  [0.99, 0.85, 0.72, 0.61, 0.51, 0.42],
                  [1.01, 0.87, 0.74, 0.63, 0.53, 0.44],
                  [0.99, 0.85, 0.72, 0.61, 0.51, 0.42]], axis=0),
    1.5: np.mean([[0.81, 0.7, 0.59, 0.5, 0.42, 0.35],
                  [0.79, 0.68, 0.58, 0.49, 0.41, 0.34],
                  [0.8, 0.69, 0.59, 0.5, 0.42, 0.34],
                  [0.82, 0.71, 0.6, 0.51, 0.43, 0.36],
                  [0.8, 0.69, 0.58, 0.49, 0.41, 0.34]], axis=0)
}

# Original trial data for each pressure
trial_data = {
    6.5: [[1.52, 1.32, 1.14, 0.98, 0.83, 0.7],
          [1.5, 1.3, 1.12, 0.96, 0.82, 0.68],
          [1.49, 1.29, 1.11, 0.95, 0.81, 0.67],
          [1.53, 1.33, 1.15, 0.99, 0.84, 0.71],
          [1.51, 1.31, 1.13, 0.97, 0.83, 0.69]],
    5.5: [[1.43, 1.24, 1.07, 0.92, 0.78, 0.65],
          [1.41, 1.22, 1.05, 0.9, 0.76, 0.63],
          [1.42, 1.23, 1.06, 0.91, 0.77, 0.64],
          [1.44, 1.25, 1.08, 0.93, 0.79, 0.66],
          [1.42, 1.23, 1.06, 0.91, 0.77, 0.64]],
    4.5: [[1.32, 1.14, 0.98, 0.84, 0.71, 0.59],
          [1.3, 1.12, 0.96, 0.82, 0.69, 0.57],
          [1.31, 1.13, 0.97, 0.83, 0.7, 0.58],
          [1.33, 1.15, 0.99, 0.85, 0.72, 0.6],
          [1.31, 1.13, 0.97, 0.83, 0.7, 0.58]],
    3.5: [[1.18, 1.02, 0.87, 0.74, 0.62, 0.51],
          [1.16, 1, 0.85, 0.72, 0.6, 0.49],
          [1.17, 1.01, 0.86, 0.73, 0.61, 0.5],
          [1.19, 1.03, 0.88, 0.75, 0.63, 0.52],
          [1.17, 1.01, 0.86, 0.73, 0.61, 0.5]],
    2.5: [[1, 0.86, 0.73, 0.62, 0.52, 0.43],
          [0.98, 0.84, 0.71, 0.6, 0.5, 0.41],
          [0.99, 0.85, 0.72, 0.61, 0.51, 0.42],
          [1.01, 0.87, 0.74, 0.63, 0.53, 0.44],
          [0.99, 0.85, 0.72, 0.61, 0.51, 0.42]],
    1.5: [[0.81, 0.7, 0.59, 0.5, 0.42, 0.35],
          [0.79, 0.68, 0.58, 0.49, 0.41, 0.34],
          [0.8, 0.69, 0.59, 0.5, 0.42, 0.34],
          [0.82, 0.71, 0.6, 0.51, 0.43, 0.36],
          [0.8, 0.69, 0.58, 0.49, 0.41, 0.34]]
}

# Calculate average rebounds for each pressure from the trial data
avg_rebounds = []
for pressure in pressures_psi:
    # Calculate mean of all heights for each pressure across all bounces and trials
    avg_rebound = np.mean([np.mean(trial) for trial in trial_data[pressure]])
    avg_rebounds.append(avg_rebound)

# Calculate average height uncertainties for first plot
avg_height_uncertainties = []
for pressure in pressures_psi:
    # Calculate mean uncertainty across all bounces
    uncertainties = [(max([trial[i] for trial in trial_data[pressure]]) - 
                     min([trial[i] for trial in trial_data[pressure]])) / 2 
                    for i in range(6)]
    avg_height_uncertainties.append(np.mean(uncertainties))

# Function to calculate COR fractional uncertainty
def calculate_cor_fractional_uncertainty(h_final, h_initial, dh_final, dh_initial):
    return 0.5 * (dh_final/h_final + dh_initial/h_initial)

# Calculate COR fractional uncertainties
cor_fractional_uncertainties_data = []
h_initial = 1.7
dh_initial = 0.020  # Initial height uncertainty

for pressure in pressures_psi:
    row_data = {'Pressure (PSI)': pressure}
    avg_heights = np.mean(trial_data[pressure], axis=0)
    
    # Get uncertainties for each height
    uncertainties = []
    for bounce_num in range(6):
        heights_at_bounce = [trial[bounce_num] for trial in trial_data[pressure]]
        uncertainty = (max(heights_at_bounce) - min(heights_at_bounce)) / 2
        uncertainties.append(uncertainty)
    
    # Calculate COR fractional uncertainty for Bounce 0-1
    row_data['Bounce 0-1'] = calculate_cor_fractional_uncertainty(
        avg_heights[0], h_initial,
        uncertainties[0], dh_initial
    )
    
    # Calculate remaining COR fractional uncertainties
    for i in range(len(avg_heights)-1):
        cor_fractional_uncertainty = calculate_cor_fractional_uncertainty(
            avg_heights[i+1], avg_heights[i],
            uncertainties[i+1], uncertainties[i]
        )
        row_data[f'Bounce {i+1}-{i+2}'] = cor_fractional_uncertainty
    
    cor_fractional_uncertainties_data.append(row_data)

# Create DataFrame for COR fractional uncertainties
df_cor_fractional_uncertainties = pd.DataFrame(cor_fractional_uncertainties_data)

# Calculate COR absolute uncertainties
cor_absolute_uncertainties_data = []
for pressure in pressures_psi:
    row_data = {'Pressure (PSI)': pressure}
    
    # Get the COR values for this pressure
    avg_heights = np.mean(trial_data[pressure], axis=0)
    initial_cor = np.sqrt(avg_heights[0] / h_initial)
    remaining_cors = np.mean([calculate_cor(trial) for trial in trial_data[pressure]], axis=0)
    
    # Get the fractional uncertainties for this pressure
    fractional_data = next(item for item in cor_fractional_uncertainties_data if item['Pressure (PSI)'] == pressure)
    
    # Calculate absolute uncertainty for Bounce 0-1
    row_data['Bounce 0-1'] = fractional_data['Bounce 0-1'] * initial_cor
    
    # Calculate remaining absolute uncertainties
    for bounce_num, cor in enumerate(remaining_cors, 1):
        bounce_key = f'Bounce {bounce_num}-{bounce_num+1}'
        row_data[bounce_key] = fractional_data[bounce_key] * cor
    
    cor_absolute_uncertainties_data.append(row_data)

# Create DataFrame for COR absolute uncertainties
df_cor_absolute_uncertainties = pd.DataFrame(cor_absolute_uncertainties_data)

# COR Analysis with uncertainties
cor_analysis_results = []

# Create initial COR data dictionary
cor_data = {
    "Pressure (PSI)": pressures_psi,
    "Bounce 0-1": [],
    "Bounce 1-2": [],
    "Bounce 2-3": [],
    "Bounce 3-4": [],
    "Bounce 4-5": [],
    "Bounce 5-6": []
}

# Calculate CORs for each pressure and bounce
for pressure in pressures_psi:
    # Calculate initial COR (Bounce 0-1)
    avg_height = np.mean([trial[0] for trial in trial_data[pressure]])
    cor_data["Bounce 0-1"].append(np.sqrt(avg_height / h0))
    
    # Calculate subsequent CORs
    avg_heights = np.mean(trial_data[pressure], axis=0)
    for i in range(len(avg_heights)-1):
        cor = np.sqrt(avg_heights[i+1] / avg_heights[i])
        cor_data[f"Bounce {i+1}-{i+2}"].append(cor)

# Convert to DataFrame and calculate uncertainties
df_cor_analysis = pd.DataFrame(cor_data)

# Calculate the mean COR for each pressure
df_cor_analysis["Average COR"] = df_cor_analysis.iloc[:, 1:].mean(axis=1)

# Calculate the standard deviation (sample standard deviation, n-1)
df_cor_analysis["Standard Deviation"] = df_cor_analysis.iloc[:, 1:7].std(axis=1, ddof=1)

# Calculate the absolute uncertainty (Standard Error of the Mean)
df_cor_analysis["Absolute Uncertainty"] = df_cor_analysis["Standard Deviation"] / np.sqrt(6)

# Calculate the fractional uncertainty
df_cor_analysis["Fractional Uncertainty"] = df_cor_analysis["Absolute Uncertainty"] / df_cor_analysis["Average COR"]

# Sort by pressure from highest to lowest
df_cor_analysis = df_cor_analysis.sort_values(by="Pressure (PSI)", ascending=False)

# Round the results
df_cor_analysis = df_cor_analysis.round({
    "Average COR": 4,
    "Standard Deviation": 4,
    "Absolute Uncertainty": 4,
    "Fractional Uncertainty": 4
})

# Calculate COR for all trials and pressures
data = []
h_initial = 1.7
for pressure in pressures_psi:
    for trial in range(5):
        heights = trial_data[pressure][trial]
        # Calculate initial COR (Bounce 0-1)
        initial_cor = np.sqrt(heights[0] / h_initial)
        # Calculate remaining CORs
        cors = calculate_cor(heights)
        
        # Add initial COR to data
        data.append({
            'Pressure (PSI)': pressure,
            'Trial': trial + 1,
            'Bounce': 'Bounce 0-1',
            'COR': initial_cor
        })
        
        # Add remaining CORs
        for bounce_num, cor in enumerate(cors, 1):
            data.append({
                'Pressure (PSI)': pressure,
                'Trial': trial + 1,
                'Bounce': f'Bounce {bounce_num}-{bounce_num+1}',
                'COR': cor
            })

# Create DataFrame for COR by trial
df = pd.DataFrame(data)
df_pivot = df.pivot_table(
    values='COR',
    index=['Pressure (PSI)', 'Trial'],
    columns='Bounce',
    aggfunc='first'
).reset_index()

# Calculate and export average heights
avg_heights_data = []
bounce_columns = [f'Bounce {i}' for i in range(1, 7)]

for pressure in pressures_psi:
    # Calculate average height across all trials for each bounce
    avg_heights = np.mean(trial_data[pressure], axis=0)
    row_data = {'Pressure (PSI)': pressure}
    for bounce_num, height in enumerate(avg_heights, 1):
        row_data[f'Bounce {bounce_num}'] = height
    avg_heights_data.append(row_data)

# Create and export average heights DataFrame
df_avg_heights = pd.DataFrame(avg_heights_data)

# Create raw trial heights DataFrame
raw_heights_data = []
for pressure in pressures_psi:
    for trial_num in range(5):
        row_data = {
            'Pressure (PSI)': pressure,
            'Trial': trial_num + 1
        }
        # Add each bounce height
        for bounce_num, height in enumerate(trial_data[pressure][trial_num], 1):
            row_data[f'Bounce {bounce_num}'] = height
        raw_heights_data.append(row_data)

# Create DataFrame with raw heights
df_raw_heights = pd.DataFrame(raw_heights_data)

# Calculate height uncertainties
height_uncertainties_data = []
for pressure in pressures_psi:
    row_data = {'Pressure (PSI)': pressure}
    
    # For each bounce number
    for bounce_num in range(6):  # 6 bounces (0-5)
        # Get all heights for this bounce across trials
        heights_at_bounce = [trial[bounce_num] for trial in trial_data[pressure]]
        # Calculate uncertainty as (max - min)/2
        uncertainty = (max(heights_at_bounce) - min(heights_at_bounce)) / 2
        row_data[f'Bounce {bounce_num + 1}'] = uncertainty
    
    height_uncertainties_data.append(row_data)

# Create DataFrame for uncertainties
df_height_uncertainties = pd.DataFrame(height_uncertainties_data)

# Calculate fractional height uncertainties
fractional_uncertainties_data = []
for pressure in pressures_psi:
    row_data = {'Pressure (PSI)': pressure}
    avg_heights = np.mean(trial_data[pressure], axis=0)
    
    # For each bounce number
    for bounce_num in range(6):
        # Get absolute uncertainty
        heights_at_bounce = [trial[bounce_num] for trial in trial_data[pressure]]
        abs_uncertainty = (max(heights_at_bounce) - min(heights_at_bounce)) / 2
        # Calculate fractional uncertainty (absolute uncertainty / magnitude)
        fractional_uncertainty = abs_uncertainty / avg_heights[bounce_num]
        row_data[f'Bounce {bounce_num + 1}'] = fractional_uncertainty
    
    fractional_uncertainties_data.append(row_data)

# Create DataFrame for fractional uncertainties
df_fractional_uncertainties = pd.DataFrame(fractional_uncertainties_data)

# Create dictionary for Excel files
excel_files = {
    'COR_by_trial': {
        'df': df_pivot,
        'filename': 'coefficient_of_restitution_by_trial.xlsx'
    },
    'average_heights': {
        'df': [df_avg_heights, df_height_uncertainties, df_fractional_uncertainties],
        'filename': 'average_heights_by_pressure.xlsx',
        'sheet_names': ['Average Heights', 'Height Uncertainties', 'Fractional Uncertainties']
    },
    'raw_heights': {
        'df': df_raw_heights,
        'filename': 'raw_rebound_heights.xlsx'
    },
    'cor_averages': {
        'df': df_cor_analysis,
        'filename': 'average_cor_with_uncertainties.xlsx'
    }
}

# Ask user for specific graph or normal execution
graph_choice = input("Enter a graph number (1-8) to see only that graph, or 'n' to continue normally: ").lower().strip()

if graph_choice.isdigit() and 1 <= int(graph_choice) <= 8:
    # Show only the selected graph
    graph_num = int(graph_choice)
    save_single = input(f"Do you want to save graph {graph_num}? (y/n): ").lower().strip() == 'y'
    
    def create_and_save_single_graph(graph_num):
        # Don't create a separate figure at the start
        # Remove: fig = plt.figure(figsize=(15.18, 7.38))
        
        if graph_num == 1:
            # Pressure vs Height plot with polynomial fits
            fig = plt.figure(figsize=(15.18, 7.38))
            for bounce_num in range(6):
                bounce_heights = []
                bounce_uncertainties = []
                for pressure in pressures_psi:
                    heights_at_bounce = [trial[bounce_num] for trial in trial_data[pressure]]
                    avg_height = np.mean(heights_at_bounce)
                    bounce_heights.append(avg_height)
                    uncertainty = (max(heights_at_bounce) - min(heights_at_bounce)) / 2
                    bounce_uncertainties.append(uncertainty)
                
                # Add polynomial fit and create label with equation
                z = np.polyfit(pressures_psi, bounce_heights, 2)
                p = np.poly1d(z)
                equation = f'y = {z[0]:.3f}x² + {z[1]:.3f}x + {z[2]:.3f}'
                
                # Plot data and fit with equation in label
                line = plt.errorbar(pressures_psi, bounce_heights, yerr=bounce_uncertainties,
                                  fmt='o', label=f'Bounce {bounce_num + 1}\n{equation}',
                                  markersize=3, capsize=3, capthick=1, elinewidth=1)
                x_fit = np.linspace(min(pressures_psi), max(pressures_psi), 100)
                plt.plot(x_fit, p(x_fit), '--', alpha=0.5, color=line.lines[0].get_color())
            plt.title('Pressure vs. Average Rebound Height per Bounce with Polynomial Fits')
            if save_single:
                fig.savefig(os.path.join(graphs_folder, 'pressure_vs_height_per_bounce.png'), dpi=100, bbox_inches='tight')
            plt.show()
            plt.close(fig)  # Close the figure after showing
                
        elif graph_num == 2:
            # Bounce vs Height plot
            fig = plt.figure(figsize=(15.18, 7.38))
            for p in pressures_psi:
                uncertainties = [(max([trial[i] for trial in trial_data[p]]) - 
                                min([trial[i] for trial in trial_data[p]])) / 2 
                                for i in range(6)]
                heights = rebound_heights[p]
                
                # Add exponential fit and create label with equation
                z = np.polyfit(bounces, np.log(heights), 1)
                equation = f'y = {np.exp(z[1]):.3f}e^({z[0]:.3f}x)'
                
                # Plot data and fit with equation in label
                line = plt.errorbar(bounces, heights, yerr=uncertainties,
                                  fmt='o', label=f'{p} PSI\n{equation}',
                                  markersize=3, capsize=3, capthick=1, elinewidth=1)
                fit_fn = lambda x: np.exp(z[1]) * np.exp(z[0] * x)
                x_fit = np.linspace(min(bounces), max(bounces), 100)
                plt.plot(x_fit, fit_fn(x_fit), '--', alpha=0.5, color=line.lines[0].get_color())
            plt.title('Bounce Number vs. Average Rebound Height with Exponential Fits')
            if save_single:
                fig.savefig(os.path.join(graphs_folder, 'bounce_vs_height.png'), dpi=100, bbox_inches='tight')
            plt.show()
            plt.close(fig)  # Close the figure after showing
                
        elif graph_num == 3:
            # Bounce vs ln(Height) plot
            fig = plt.figure(figsize=(15.18, 7.38))
            for p in pressures_psi:
                heights = rebound_heights[p]
                uncertainties = [(max([trial[i] for trial in trial_data[p]]) - 
                                min([trial[i] for trial in trial_data[p]])) / 2 
                                for i in range(6)]
                log_uncertainties = [u/h for u, h in zip(uncertainties, heights)]
                
                # Add linear fit and create label with equation
                z = np.polyfit(bounces, np.log(heights), 1)
                equation = f'ln(y) = {z[0]:.3f}x + {z[1]:.3f}'
                
                # Plot data and fit with equation in label
                line = plt.errorbar(bounces, np.log(heights), yerr=log_uncertainties,
                                  fmt='o', label=f'{p} PSI\n{equation}',
                                  markersize=3, capsize=3, capthick=1, elinewidth=1)
                p_fit = np.poly1d(z)
                x_fit = np.linspace(min(bounces), max(bounces), 100)
                plt.plot(x_fit, p_fit(x_fit), '--', alpha=0.5, color=line.lines[0].get_color())
            plt.title('Bounce Number vs. ln(Rebound Height) with Linear Fits')
            if save_single:
                fig.savefig(os.path.join(graphs_folder, 'bounce_vs_ln_height.png'), dpi=100, bbox_inches='tight')
            plt.show()
            plt.close(fig)  # Close the figure after showing
                
        elif graph_num == 4:
            # Bounce vs COR plot
            fig = plt.figure(figsize=(15.18, 7.38))
            for p in pressures_psi:
                cor_values = calculate_cor(rebound_heights[p])
                abs_uncertainties = df_cor_absolute_uncertainties[df_cor_absolute_uncertainties['Pressure (PSI)'] == p].iloc[0, 1:].values
                
                # Add exponential fit and create label with equation
                z = np.polyfit(bounces[1:], np.log(cor_values), 1)
                equation = f'y = {np.exp(z[1]):.3f}e^({z[0]:.3f}x)'
                
                # Plot data and fit with equation in label
                line = plt.errorbar(bounces[1:], cor_values, yerr=abs_uncertainties[1:],
                                  fmt='o', label=f'{p} PSI\n{equation}',
                                  markersize=3, capsize=3, capthick=1, elinewidth=1)
                fit_fn = lambda x: np.exp(z[1]) * np.exp(z[0] * x)
                x_fit = np.linspace(min(bounces[1:]), max(bounces[1:]), 100)
                plt.plot(x_fit, fit_fn(x_fit), '--', alpha=0.5, color=line.lines[0].get_color())
            plt.title('Bounce Number vs. Coefficient of Restitution with Exponential Fits')
            if save_single:
                fig.savefig(os.path.join(graphs_folder, 'bounce_vs_cor.png'), dpi=100, bbox_inches='tight')
            plt.show()
            plt.close(fig)  # Close the figure after showing
                
        elif graph_num == 5:
            # Pressure vs Initial COR plot
            fig5a = plt.figure(figsize=(15.18, 7.38))
            
            # Get data for initial bounce (0-1)
            cors = []
            uncertainties = []
            
            for pressure in pressures_psi:
                # For 0-1 bounce, use initial height of 1.7m
                avg_height = np.mean([trial[0] for trial in trial_data[pressure]])
                cor = np.sqrt(avg_height / 1.7)
                uncertainty = df_cor_absolute_uncertainties[
                    df_cor_absolute_uncertainties['Pressure (PSI)'] == pressure
                ]['Bounce 0-1'].values[0]
                
                cors.append(cor)
                uncertainties.append(uncertainty)
            
            # Add logarithmic fit and create label with equation
            z = np.polyfit(np.log(pressures_psi), cors, 1)
            equation = f'y = {z[0]:.3f}ln(x) + {z[1]:.3f}'
            
            # Plot data, error bars, and fit with matching colors
            line = plt.errorbar(pressures_psi, cors, yerr=uncertainties,
                        fmt='o', label=f'Initial Bounce (0-1)\n{equation}',
                        markersize=3, capsize=3, capthick=1, elinewidth=1)
            color = line.lines[0].get_color()
            
            # Set the error bar color to match the points
            line[1][0].set_color(color)
            line[2][0].set_color(color)
            
            # Plot logarithmic fit with matching color
            x_fit = np.linspace(min(pressures_psi), max(pressures_psi), 100)
            plt.plot(x_fit, z[0] * np.log(x_fit) + z[1], '--', alpha=0.5, color=color)

            plt.xlabel('Internal Pressure (PSI)')
            plt.ylabel('Coefficient of Restitution (COR)')
            plt.title('Pressure vs. Initial Coefficient of Restitution (Bounce 0-1)')
            plt.legend()
            plt.grid(True)
            if save_single:
                fig.savefig(os.path.join(graphs_folder, 'pressure_vs_initial_cor.png'), dpi=100, bbox_inches='tight')
            plt.show()
            plt.close(fig)  # Close the figure after showing
                
        elif graph_num == 6:
            # Combined Pressure vs All COR plot
            fig6 = plt.figure(figsize=(15.18, 7.38))
            
            # First plot initial bounce (0-1)
            cors = []
            uncertainties = []
            for pressure in pressures_psi:
                avg_height = np.mean([trial[0] for trial in trial_data[pressure]])
                cor = np.sqrt(avg_height / 1.7)
                uncertainty = df_cor_absolute_uncertainties[
                    df_cor_absolute_uncertainties['Pressure (PSI)'] == pressure
                ]['Bounce 0-1'].values[0]
                cors.append(cor)
                uncertainties.append(uncertainty)
            
            z = np.polyfit(np.log(pressures_psi), cors, 1)
            equation = f'y = {z[0]:.3f}ln(x) + {z[1]:.3f}'
            
            line = plt.errorbar(pressures_psi, cors, yerr=uncertainties,
                        fmt='o', label=f'Initial Bounce (0-1)\n{equation}',
                        markersize=3, capsize=3, capthick=1, elinewidth=1)
            color = line.lines[0].get_color()
            line[1][0].set_color(color)
            line[2][0].set_color(color)
            
            x_fit = np.linspace(min(pressures_psi), max(pressures_psi), 100)
            plt.plot(x_fit, z[0] * np.log(x_fit) + z[1], '--', alpha=0.5, color=color)

            # Then plot subsequent bounces (1-2 through 5-6)
            for bounce_transition in range(1, 6):
                cors = []
                uncertainties = []
                for pressure in pressures_psi:
                    avg_heights = np.mean(trial_data[pressure], axis=0)
                    cor = np.sqrt(avg_heights[bounce_transition] / avg_heights[bounce_transition - 1])
                    uncertainty = df_cor_absolute_uncertainties[
                        df_cor_absolute_uncertainties['Pressure (PSI)'] == pressure
                    ][f'Bounce {bounce_transition}-{bounce_transition+1}'].values[0]
                    cors.append(cor)
                    uncertainties.append(uncertainty)
                
                z = np.polyfit(np.log(pressures_psi), cors, 1)
                equation = f'y = {z[0]:.3f}ln(x) + {z[1]:.3f}'
                
                line = plt.errorbar(pressures_psi, cors, yerr=uncertainties,
                                  fmt='o', label=f'Bounce {bounce_transition}-{bounce_transition+1}\n{equation}',
                                  markersize=3, capsize=3, capthick=1, elinewidth=1)
                color = line.lines[0].get_color()
                line[1][0].set_color(color)
                line[2][0].set_color(color)
                
                x_fit = np.linspace(min(pressures_psi), max(pressures_psi), 100)
                plt.plot(x_fit, z[0] * np.log(x_fit) + z[1], '--', alpha=0.5, color=color)
            plt.title('Pressure vs. All Coefficients of Restitution')
            if save_single:
                fig.savefig(os.path.join(graphs_folder, 'pressure_vs_all_cor.png'), dpi=100, bbox_inches='tight')
            plt.show()
            plt.close(fig)  # Close the figure after showing
                
        elif graph_num == 7:
            # Average COR vs Pressure plot
            fig = plt.figure(figsize=(15.18, 7.38))
            
            # Get data from df_cor_analysis
            pressures = df_cor_analysis['Pressure (PSI)'].values
            avg_cors = df_cor_analysis['Average COR'].values
            abs_uncertainties = df_cor_analysis['Absolute Uncertainty'].values
            
            # Add logarithmic fit and create label with equation
            z = np.polyfit(np.log(pressures), avg_cors, 1)
            equation = f'y = {z[0]:.3f}ln(x) + {z[1]:.3f}'
            
            # Plot data with error bars using new uncertainties
            line = plt.errorbar(pressures, avg_cors, yerr=abs_uncertainties,
                        fmt='o', label=f'Average COR\n{equation}',
                        markersize=5, capsize=3, capthick=1, elinewidth=1)
            
            # Plot logarithmic fit
            x_fit = np.linspace(min(pressures), max(pressures), 100)
            plt.plot(x_fit, z[0] * np.log(x_fit) + z[1], '--', alpha=0.5, 
                     color=line.lines[0].get_color())

            plt.xlabel('Internal Pressure (PSI)')
            plt.ylabel('Average Coefficient of Restitution (COR)')
            plt.title('Pressure vs. Average COR with Standard Error of the Mean')
            plt.legend()
            plt.grid(True)
            if save_single:
                fig.savefig(os.path.join(graphs_folder, 'pressure_vs_average_cor.png'), dpi=100, bbox_inches='tight')
            plt.show()
            plt.close(fig)  # Close the figure after showing
        elif graph_num == 8:
            # Linearized Average COR vs ln(Pressure) plot
            fig8 = plt.figure(figsize=(15.18, 7.38))
            
            # Get data from df_cor_analysis
            pressures = df_cor_analysis['Pressure (PSI)'].values
            avg_cors = df_cor_analysis['Average COR'].values
            
            # Calculate ln(pressure) values
            ln_pressures = np.log(pressures)
            
            # Get fractional uncertainties from df_cor_analysis
            fractional_uncertainties = df_cor_analysis['Fractional Uncertainty'].values
            yerr = avg_cors * fractional_uncertainties
            
            # Calculate the best fit line
            z = np.polyfit(ln_pressures, avg_cors, 1)
            equation = f'y = {z[0]:.3f}x + {z[1]:.3f}'
            r_squared = np.corrcoef(ln_pressures, avg_cors)[0,1]**2
            
            # Calculate maximum and minimum slope lines
            # Maximum slope line through (x1,y1+dy1) and (x2,y2-dy2)
            x1, x2 = ln_pressures[0], ln_pressures[-1]
            y1, y2 = avg_cors[0], avg_cors[-1]
            dy1, dy2 = yerr[0], yerr[-1]
            
            max_slope = ((y2 + dy2) - (y1 - dy1)) / (x2 - x1)
            max_intercept = (y1 - dy1) - max_slope * x1
            max_equation = f'y = {max_slope:.3f}x + {max_intercept:.3f}'
            
            # Minimum slope line through (x1,y1-dy1) and (x2,y2+dy2)
            min_slope = ((y2 - dy2) - (y1 + dy1)) / (x2 - x1)
            min_intercept = (y1 + dy1) - min_slope * x1
            min_equation = f'y = {min_slope:.3f}x + {min_intercept:.3f}'
            
            # Plot data with fractional uncertainties as error bars
            line = plt.errorbar(ln_pressures, avg_cors, yerr=yerr,
                        fmt='o', label=f'Average COR\nBest fit: {equation}\nR² = {r_squared:.3f}',
                        markersize=5, capsize=3, capthick=1, elinewidth=1)
            color = line.lines[0].get_color()
            
            # Plot best fit line
            x_fit = np.linspace(min(ln_pressures), max(ln_pressures), 100)
            plt.plot(x_fit, z[0] * x_fit + z[1], '--', alpha=0.5, 
                     color=color, label='Best fit')
            
            # Plot maximum slope line
            plt.plot(x_fit, max_slope * x_fit + max_intercept, ':',
                    alpha=0.5, color='red', label=f'Maximum slope: {max_equation}')
            
            # Plot minimum slope line
            plt.plot(x_fit, min_slope * x_fit + min_intercept, ':',
                    alpha=0.5, color='blue', label=f'Minimum slope: {min_equation}')

            plt.xlabel('ln(Pressure) [ln(PSI)]')
            plt.ylabel('Average Coefficient of Restitution (COR)')
            plt.title('Linearized Relationship: ln(Pressure) vs. Average COR\nwith Fractional Uncertainties')
            plt.legend()
            plt.grid(True)
            if save_single:
                fig8.savefig(os.path.join(graphs_folder, 'ln_pressure_vs_average_cor.png'), dpi=100, bbox_inches='tight')
            plt.show()
            plt.close(fig8)
    
    # Create the graphs folder if saving is requested
    if save_single:
        os.makedirs(graphs_folder, exist_ok=True)
    
    # Show and optionally save the selected graph
    create_and_save_single_graph(graph_num)
    
else:
    # Continue with normal execution
    show_graphs = input("Do you want to see the graphs? (y/n): ").lower().strip() == 'y'

    if show_graphs:
        # Ask user if they want to download the graphs
        save_graphs = input("Do you want to download the graphs? (y/n): ").lower().strip() == 'y'

        # Function to handle plotting and saving
        def plot_and_save(fig, filename):
            if save_graphs:
                save_path = os.path.join(graphs_folder, filename)
                fig.set_size_inches(15.18, 7.38)
                fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.show()
            plt.close(fig)  # Close the figure after showing

        # 1. Pressure vs Height plot with polynomial fits
        fig1 = plt.figure(figsize=(15.18, 7.38))
        for bounce_num in range(6):
            bounce_heights = []
            bounce_uncertainties = []
            for pressure in pressures_psi:
                heights_at_bounce = [trial[bounce_num] for trial in trial_data[pressure]]
                avg_height = np.mean(heights_at_bounce)
                bounce_heights.append(avg_height)
                uncertainty = (max(heights_at_bounce) - min(heights_at_bounce)) / 2
                bounce_uncertainties.append(uncertainty)
            
            # Add polynomial fit and create label with equation
            z = np.polyfit(pressures_psi, bounce_heights, 2)
            p = np.poly1d(z)
            equation = f'y = {z[0]:.3f}x² + {z[1]:.3f}x + {z[2]:.3f}'
            
            # Plot data and fit with equation in label
            line = plt.errorbar(pressures_psi, bounce_heights, yerr=bounce_uncertainties,
                              fmt='o', label=f'Bounce {bounce_num + 1}\n{equation}',
                              markersize=3, capsize=3, capthick=1, elinewidth=1)
            x_fit = np.linspace(min(pressures_psi), max(pressures_psi), 100)
            plt.plot(x_fit, p(x_fit), '--', alpha=0.5, color=line.lines[0].get_color())

        plt.xlabel('Internal Pressure (PSI)')
        plt.ylabel('Average Rebound Height (m)')
        plt.title('Pressure vs. Average Rebound Height per Bounce with Polynomial Fits')
        plt.legend()
        plt.grid(True)
        plot_and_save(fig1, 'pressure_vs_height_per_bounce.png')

        # 2. Bounce vs Height plot with exponential fits
        fig2 = plt.figure(figsize=(15.18, 7.38))
        for p in pressures_psi:
            uncertainties = [(max([trial[i] for trial in trial_data[p]]) - 
                            min([trial[i] for trial in trial_data[p]])) / 2 
                            for i in range(6)]
            heights = rebound_heights[p]
            
            # Add exponential fit and create label with equation
            z = np.polyfit(bounces, np.log(heights), 1)
            equation = f'y = {np.exp(z[1]):.3f}e^({z[0]:.3f}x)'
            
            # Plot data and fit with equation in label
            line = plt.errorbar(bounces, heights, yerr=uncertainties,
                              fmt='o', label=f'{p} PSI\n{equation}',
                              markersize=3, capsize=3, capthick=1, elinewidth=1)
            fit_fn = lambda x: np.exp(z[1]) * np.exp(z[0] * x)
            x_fit = np.linspace(min(bounces), max(bounces), 100)
            plt.plot(x_fit, fit_fn(x_fit), '--', alpha=0.5, color=line.lines[0].get_color())

        plt.xlabel('Bounce Number')
        plt.ylabel('Average Rebound Height (m)')
        plt.title('Bounce Number vs. Average Rebound Height with Exponential Fits')
        plt.legend(title='Pressure')
        plt.grid(True)
        plot_and_save(fig2, 'bounce_vs_height.png')

        # 3. Bounce vs ln(Height) plot with linear fits
        fig3 = plt.figure(figsize=(15.18, 7.38))
        for p in pressures_psi:
            heights = rebound_heights[p]
            uncertainties = [(max([trial[i] for trial in trial_data[p]]) - 
                            min([trial[i] for trial in trial_data[p]])) / 2 
                            for i in range(6)]
            log_uncertainties = [u/h for u, h in zip(uncertainties, heights)]
            
            # Add linear fit and create label with equation
            z = np.polyfit(bounces, np.log(heights), 1)
            equation = f'ln(y) = {z[0]:.3f}x + {z[1]:.3f}'
            
            # Plot data and fit with equation in label
            line = plt.errorbar(bounces, np.log(heights), yerr=log_uncertainties,
                              fmt='o', label=f'{p} PSI\n{equation}',
                              markersize=3, capsize=3, capthick=1, elinewidth=1)
            p_fit = np.poly1d(z)
            x_fit = np.linspace(min(bounces), max(bounces), 100)
            plt.plot(x_fit, p_fit(x_fit), '--', alpha=0.5, color=line.lines[0].get_color())

        plt.xlabel('Bounce Number')
        plt.ylabel('ln(Average Rebound Height)')
        plt.title('Bounce Number vs. ln(Rebound Height) with Linear Fits')
        plt.legend(title='Pressure')
        plt.grid(True)
        plot_and_save(fig3, 'bounce_vs_ln_height.png')

        # 4. Bounce vs COR plot with exponential fits
        fig4 = plt.figure(figsize=(15.18, 7.38))
        for p in pressures_psi:
            cor_values = calculate_cor(rebound_heights[p])
            abs_uncertainties = df_cor_absolute_uncertainties[df_cor_absolute_uncertainties['Pressure (PSI)'] == p].iloc[0, 1:].values
            
            # Add exponential fit and create label with equation
            z = np.polyfit(bounces[1:], np.log(cor_values), 1)
            equation = f'y = {np.exp(z[1]):.3f}e^({z[0]:.3f}x)'
            
            # Plot data and fit with equation in label
            line = plt.errorbar(bounces[1:], cor_values, yerr=abs_uncertainties[1:],
                              fmt='o', label=f'{p} PSI\n{equation}',
                              markersize=3, capsize=3, capthick=1, elinewidth=1)
            fit_fn = lambda x: np.exp(z[1]) * np.exp(z[0] * x)
            x_fit = np.linspace(min(bounces[1:]), max(bounces[1:]), 100)
            plt.plot(x_fit, fit_fn(x_fit), '--', alpha=0.5, color=line.lines[0].get_color())

        plt.xlabel('Bounce Number')
        plt.ylabel('Coefficient of Restitution (COR)')
        plt.title('Bounce Number vs. Coefficient of Restitution with Exponential Fits')
        plt.legend(title='Pressure')
        plt.grid(True)
        plot_and_save(fig4, 'bounce_vs_cor.png')

        # 5a. Pressure vs Initial COR (Bounce 0-1) plot with logarithmic fit
        fig5a = plt.figure(figsize=(15.18, 7.38))
        
        # Get data for initial bounce (0-1)
        cors = []
        uncertainties = []
        
        for pressure in pressures_psi:
            # For 0-1 bounce, use initial height of 1.7m
            avg_height = np.mean([trial[0] for trial in trial_data[pressure]])
            cor = np.sqrt(avg_height / 1.7)
            uncertainty = df_cor_absolute_uncertainties[
                df_cor_absolute_uncertainties['Pressure (PSI)'] == pressure
            ]['Bounce 0-1'].values[0]
            
            cors.append(cor)
            uncertainties.append(uncertainty)
        
        # Add logarithmic fit and create label with equation
        z = np.polyfit(np.log(pressures_psi), cors, 1)
        equation = f'y = {z[0]:.3f}ln(x) + {z[1]:.3f}'
        
        # Plot data, error bars, and fit with matching colors
        line = plt.errorbar(pressures_psi, cors, yerr=uncertainties,
                    fmt='o', label=f'Initial Bounce (0-1)\n{equation}',
                    markersize=3, capsize=3, capthick=1, elinewidth=1)
        color = line.lines[0].get_color()
        
        # Set the error bar color to match the points
        line[1][0].set_color(color)
        line[2][0].set_color(color)
        
        # Plot logarithmic fit with matching color
        x_fit = np.linspace(min(pressures_psi), max(pressures_psi), 100)
        plt.plot(x_fit, z[0] * np.log(x_fit) + z[1], '--', alpha=0.5, color=color)

        plt.xlabel('Internal Pressure (PSI)')
        plt.ylabel('Coefficient of Restitution (COR)')
        plt.title('Pressure vs. Initial Coefficient of Restitution (Bounce 0-1)')
        plt.legend()
        plt.grid(True)
        plot_and_save(fig5a, 'pressure_vs_initial_cor.png')

        # 5b. Pressure vs Subsequent COR plot with logarithmic fits
        fig5b = plt.figure(figsize=(15.18, 7.38))
        
        # Get data for each bounce transition (1-2 through 5-6)
        for bounce_transition in range(1, 6):  # 1 to 5
            cors = []
            uncertainties = []
            
            for pressure in pressures_psi:
                avg_heights = np.mean(trial_data[pressure], axis=0)
                cor = np.sqrt(avg_heights[bounce_transition] / avg_heights[bounce_transition - 1])
                uncertainty = df_cor_absolute_uncertainties[
                    df_cor_absolute_uncertainties['Pressure (PSI)'] == pressure
                ][f'Bounce {bounce_transition}-{bounce_transition+1}'].values[0]
                
                cors.append(cor)
                uncertainties.append(uncertainty)
            
            # Add logarithmic fit and create label with equation
            z = np.polyfit(np.log(pressures_psi), cors, 1)
            equation = f'y = {z[0]:.3f}ln(x) + {z[1]:.3f}'
            
            # Plot data, error bars, and fit with matching colors
            line = plt.errorbar(pressures_psi, cors, yerr=uncertainties,
                              fmt='o', label=f'Bounce {bounce_transition}-{bounce_transition+1}\n{equation}',
                              markersize=3, capsize=3, capthick=1, elinewidth=1)
            color = line.lines[0].get_color()
            
            # Set the error bar color to match the points
            line[1][0].set_color(color)
            line[2][0].set_color(color)
            
            # Plot logarithmic fit with matching color
            x_fit = np.linspace(min(pressures_psi), max(pressures_psi), 100)
            plt.plot(x_fit, z[0] * np.log(x_fit) + z[1], '--', alpha=0.5, color=color)

        plt.xlabel('Internal Pressure (PSI)')
        plt.ylabel('Coefficient of Restitution (COR)')
        plt.title('Pressure vs. Subsequent Coefficients of Restitution')
        plt.legend()
        plt.grid(True)
        plot_and_save(fig5b, 'pressure_vs_subsequent_cor.png')

        # 6. Combined Pressure vs All COR plot with logarithmic fits
        fig6 = plt.figure(figsize=(15.18, 7.38))
        
        # First plot initial bounce (0-1)
        cors = []
        uncertainties = []
        for pressure in pressures_psi:
            avg_height = np.mean([trial[0] for trial in trial_data[pressure]])
            cor = np.sqrt(avg_height / 1.7)
            uncertainty = df_cor_absolute_uncertainties[
                df_cor_absolute_uncertainties['Pressure (PSI)'] == pressure
            ]['Bounce 0-1'].values[0]
            cors.append(cor)
            uncertainties.append(uncertainty)
        
        z = np.polyfit(np.log(pressures_psi), cors, 1)
        equation = f'y = {z[0]:.3f}ln(x) + {z[1]:.3f}'
        
        line = plt.errorbar(pressures_psi, cors, yerr=uncertainties,
                    fmt='o', label=f'Initial Bounce (0-1)\n{equation}',
                    markersize=3, capsize=3, capthick=1, elinewidth=1)
        color = line.lines[0].get_color()
        line[1][0].set_color(color)
        line[2][0].set_color(color)
        
        x_fit = np.linspace(min(pressures_psi), max(pressures_psi), 100)
        plt.plot(x_fit, z[0] * np.log(x_fit) + z[1], '--', alpha=0.5, color=color)

        # Then plot subsequent bounces (1-2 through 5-6)
        for bounce_transition in range(1, 6):
            cors = []
            uncertainties = []
            for pressure in pressures_psi:
                avg_heights = np.mean(trial_data[pressure], axis=0)
                cor = np.sqrt(avg_heights[bounce_transition] / avg_heights[bounce_transition - 1])
                uncertainty = df_cor_absolute_uncertainties[
                    df_cor_absolute_uncertainties['Pressure (PSI)'] == pressure
                ][f'Bounce {bounce_transition}-{bounce_transition+1}'].values[0]
                cors.append(cor)
                uncertainties.append(uncertainty)
            
            z = np.polyfit(np.log(pressures_psi), cors, 1)
            equation = f'y = {z[0]:.3f}ln(x) + {z[1]:.3f}'
            
            line = plt.errorbar(pressures_psi, cors, yerr=uncertainties,
                              fmt='o', label=f'Bounce {bounce_transition}-{bounce_transition+1}\n{equation}',
                              markersize=3, capsize=3, capthick=1, elinewidth=1)
            color = line.lines[0].get_color()
            line[1][0].set_color(color)
            line[2][0].set_color(color)
            
            x_fit = np.linspace(min(pressures_psi), max(pressures_psi), 100)
            plt.plot(x_fit, z[0] * np.log(x_fit) + z[1], '--', alpha=0.5, color=color)

        plt.xlabel('Internal Pressure (PSI)')
        plt.ylabel('Coefficient of Restitution (COR)')
        plt.title('Pressure vs. All Coefficients of Restitution')
        plt.legend()
        plt.grid(True)
        plot_and_save(fig6, 'pressure_vs_all_cor.png')

        # 7. Average COR vs Pressure plot with logarithmic fit
        fig7 = plt.figure(figsize=(15.18, 7.38))
        
        # Get data from df_cor_analysis
        pressures = df_cor_analysis['Pressure (PSI)'].values
        avg_cors = df_cor_analysis['Average COR'].values
        abs_uncertainties = df_cor_analysis['Absolute Uncertainty'].values
        
        # Add logarithmic fit and create label with equation
        z = np.polyfit(np.log(pressures), avg_cors, 1)
        equation = f'y = {z[0]:.3f}ln(x) + {z[1]:.3f}'
        
        # Plot data with error bars using new uncertainties
        line = plt.errorbar(pressures, avg_cors, yerr=abs_uncertainties,
                    fmt='o', label=f'Average COR\n{equation}\nTypical fractional uncertainty: {df_cor_analysis["Fractional Uncertainty"].mean():.3%}',
                    markersize=5, capsize=3, capthick=1, elinewidth=1)
        
        # Plot logarithmic fit
        x_fit = np.linspace(min(pressures), max(pressures), 100)
        plt.plot(x_fit, z[0] * np.log(x_fit) + z[1], '--', alpha=0.5, 
                 color=line.lines[0].get_color())

        plt.xlabel('Internal Pressure (PSI)')
        plt.ylabel('Average Coefficient of Restitution (COR)')
        plt.title('Pressure vs. Average COR with Standard Error of the Mean')
        plt.legend()
        plt.grid(True)
        plot_and_save(fig7, 'pressure_vs_average_cor.png')

        # 8. Linearized Average COR vs ln(Pressure) plot
        fig8 = plt.figure(figsize=(15.18, 7.38))
        
        # Calculate ln(pressure) values
        ln_pressures = np.log(pressures)
        
        # Get fractional uncertainties from df_cor_analysis
        fractional_uncertainties = df_cor_analysis['Fractional Uncertainty'].values
        yerr = avg_cors * fractional_uncertainties
        
        # Calculate the best fit line
        z = np.polyfit(ln_pressures, avg_cors, 1)
        equation = f'y = {z[0]:.3f}x + {z[1]:.3f}'
        r_squared = np.corrcoef(ln_pressures, avg_cors)[0,1]**2
        
        # Calculate maximum and minimum slope lines
        # Maximum slope line through (x1,y1+dy1) and (x2,y2-dy2)
        x1, x2 = ln_pressures[0], ln_pressures[-1]
        y1, y2 = avg_cors[0], avg_cors[-1]
        dy1, dy2 = yerr[0], yerr[-1]
        
        max_slope = ((y2 + dy2) - (y1 - dy1)) / (x2 - x1)
        max_intercept = (y1 - dy1) - max_slope * x1
        max_equation = f'y = {max_slope:.3f}x + {max_intercept:.3f}'
        
        # Minimum slope line through (x1,y1-dy1) and (x2,y2+dy2)
        min_slope = ((y2 - dy2) - (y1 + dy1)) / (x2 - x1)
        min_intercept = (y1 + dy1) - min_slope * x1
        min_equation = f'y = {min_slope:.3f}x + {min_intercept:.3f}'
        
        # Plot data with fractional uncertainties as error bars
        line = plt.errorbar(ln_pressures, avg_cors, yerr=yerr,
                    fmt='o', label=f'Average COR\nBest fit: {equation}\nR² = {r_squared:.3f}',
                    markersize=5, capsize=3, capthick=1, elinewidth=1)
        color = line.lines[0].get_color()
        
        # Plot best fit line
        x_fit = np.linspace(min(ln_pressures), max(ln_pressures), 100)
        plt.plot(x_fit, z[0] * x_fit + z[1], '--', alpha=0.5, 
                 color=color, label='Best fit')
        
        # Plot maximum slope line
        plt.plot(x_fit, max_slope * x_fit + max_intercept, ':',
                alpha=0.5, color='red', label=f'Maximum slope: {max_equation}')
        
        # Plot minimum slope line
        plt.plot(x_fit, min_slope * x_fit + min_intercept, ':',
                alpha=0.5, color='blue', label=f'Minimum slope: {min_equation}')

        plt.xlabel('ln(Pressure) [ln(PSI)]')
        plt.ylabel('Average Coefficient of Restitution (COR)')
        plt.title('Linearized Relationship: ln(Pressure) vs. Average COR\nwith Fractional Uncertainties')
        plt.legend()
        plt.grid(True)
        plot_and_save(fig8, 'ln_pressure_vs_average_cor.png')

        if save_graphs:
            print("\nGraphs have been saved successfully.")

# Add this to the excel_files dictionary before the export loop
excel_files['cor_analysis'] = {
    'df': df_cor_analysis,
    'filename': 'cor_analysis_results.xlsx'
}

# Before exporting Excel files, ask user preference
save_excel = input("Do you want to download the Excel sheets? (y/n): ").lower().strip() == 'y'

# Export all Excel files
for file_info in excel_files.values():
    try:
        file_path = os.path.join(excel_folder, file_info['filename'])
        if isinstance(file_info['df'], list):
            with pd.ExcelWriter(file_path) as writer:
                for df, sheet_name in zip(file_info['df'], file_info['sheet_names']):
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            file_info['df'].to_excel(file_path, index=False)
    except Exception as e:
        print(f"Error during Excel file creation: {str(e)}")

# Only delete Excel files if user chose not to save them
if not save_excel:
    # Delete all Excel files and the excel_folder
    for file_info in excel_files.values():
        try:
            file_path = os.path.join(excel_folder, file_info['filename'])
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting temporary file: {str(e)}")

    # Remove the excel_folder if it's empty
    try:
        if os.path.exists(excel_folder) and not os.listdir(excel_folder):
            os.rmdir(excel_folder)
    except Exception as e:
        print(f"Error removing excel folder: {str(e)}")
elif save_excel:
    print(f"\nExcel files have been saved in: {excel_folder}")

# Only print information about graphs
print("\nAll calculations completed.")
if 'graphs_folder' in locals() and os.path.exists(graphs_folder):
    print(f"Graphs are saved in: {graphs_folder}")