import numpy as np
from scipy import stats

def calculate_linear_p_value(df_cor_analysis):
    """
    Calculate the p-value for the linear relationship between ln(Pressure) and COR
    
    Parameters:
    df_cor_analysis: DataFrame containing 'Pressure (PSI)' and 'Average COR' columns
    
    Returns:
    dict: Contains slope, intercept, r_value, p_value, and std_err
    """
    
    # Get the data from the DataFrame
    pressures = df_cor_analysis['Pressure (PSI)'].values
    avg_cors = df_cor_analysis['Average COR'].values
    
    # Calculate ln(pressure) values
    ln_pressures = np.log(pressures)
    
    # Perform linear regression and get p-value
    slope, intercept, r_value, p_value, std_err = stats.linregress(ln_pressures, avg_cors)
    
    results = {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }
    
    return results

if __name__ == "__main__":
    # Example usage:
    import pandas as pd
    
    # Load your data or use the provided df_cor_analysis
    try:
        # Assuming df_cor_analysis is available from data_analysis.py
        from data_analysis import df_cor_analysis
        
        results = calculate_linear_p_value(df_cor_analysis)
        
        print("\nStatistical Analysis of ln(Pressure) vs COR Relationship:")
        print(f"Slope: {results['slope']:.4f} ± {results['std_err']:.4f}")
        print(f"Intercept: {results['intercept']:.4f}")
        print(f"R-squared: {results['r_squared']:.4f}")
        print(f"P-value: {results['p_value']:.4e}")
        
        # Interpret the p-value
        alpha = 0.05
        if results['p_value'] < alpha:
            print(f"\nThe relationship is statistically significant (p < {alpha})")
        else:
            print(f"\nThe relationship is not statistically significant (p > {alpha})")
            
    except ImportError:
        print("Please run this after running data_analysis.py")
