import jax
import numpyro as npyro
import numpyro.distributions as dists
import numpyro.distributions.transforms as tfs
# Device count has to be set before importing jax
npyro.set_host_device_count(4)


import jax.numpy as jnp
import jax.random as rand

import time
from functools import partial
import json
import pandas as pd

import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt, scale
import matplotlib.lines as pltlines

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import stratcona


# This script is for generating data for modelling the reliability of hybrid bonding interconnects from dielectric degradation
# This specifically models SiCN reliability but this could be adapted to other dielectrics
# I am using this paper for reference: https://ieeexplore.ieee.org/abstract/document/9764478
# Paper uses the E model which states that 
# t_50 is proportional to exp(gamma * E) where E is the electric field across the dielectric and gamma the field acceleration factor
# This paper uses voltage ramp tests to determine gamma, the field acceleration factor
# Then you can use the power law to further project lifetime

# Keeping this model extremely simple so it can just be an extra mechanism

def hybrid_bonding_dielectric():
    

    ########################################################
    # Defining Equation
    ########################################################

    # This function is to help find the fit 
    # Parameters:
    # - ramp_rate: the rate at which voltage is ramped in the test (V/s)
    # - vbd: the breakdown voltage observed in the test (V)
    # - const: constant to fit the slope  (just a fitting parameter)
    def calc_gamma_fit(ramp_rate, vbd, gamma, const):
       # ln(ramp_rate) = (gamma - 1) * ln(vbd) + const
       gamma = 1 + (jnp.log(ramp_rate) - const) / jnp.log(vbd)
       return gamma
    
    
    ##################################################
    #  Data from Paper, Fig 5 
    #  Source: https://ieeexplore.ieee.org/abstract/document/9764478
    ##################################################

    # The  data is read off graphs from figure 5 of this paper which was challening to est
    
    # Data points extracted from paper Fig 5
    # Equation: ln(ramp_rate) = (gamma - 1) * ln(vbd) + const
    # Only including clearly visible points from each graph
    
    # From 100C, gamma = 11.5 - graph (a)
    vbd_ramp_rate_100C = [[173, 0.05], [195, 0.17]]

    # From 150C, gamma = 12.3 - graph (b)
    vbd_ramp_rate_150C = [[148, 0.058], [168, 0.18], [186, 1.2]]

    # From 175C, gamma = 12.4 - graph (c)
    vbd_ramp_rate_175C = [[145, 0.052], [152, 0.18], [178, 1.1]]

    # From 200C, gamma = 9.0 - graph (d)
    vbd_ramp_rate_200C = [[127, 0.05], [136, 0.18], [168, 1.1]]

    ########################################################
    # Calculate Gamma for Each Temperature
    ########################################################
    def calculate_gamma_from_data(data_points, temp_label):
        """
        Calculate gamma (field acceleration factor) from voltage ramp test data
        Using the calc_gamma_fit function: gamma = 1 + (ln(ramp_rate) - const) / ln(vbd)
        First finds the constant from linear regression, then uses calc_gamma_fit on each point
        
        Parameters:
        - data_points: list of [vbd_voltage, ramp_rate] points
        - temp_label: temperature label for reporting
        
        Returns:
        - gamma: calculated field acceleration factor (average from all points)
        """
        vbd = np.array([pt[0] for pt in data_points])
        ramp_rate = np.array([pt[1] for pt in data_points])
        
        # Linear regression in log-log space: ln(ramp_rate) vs ln(vbd)
        # to find the constant (intercept)
        x = np.log(vbd)
        y = np.log(ramp_rate)
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        const = coeffs[1]
        
        # Use calc_gamma_fit on each point
        gamma_values = []
        print(f"\n{temp_label}:")
        print(f"  Data points (Vbd, Ramp Rate): {data_points}")
        print(f"  Fitted constant: {const:.4f}")
        
        for pt in data_points:
            vbd_pt = pt[0]
            ramp_pt = pt[1]
            # Convert to JAX arrays for calc_gamma_fit
            gamma_calc = 1 + (np.log(ramp_pt) - const) / np.log(vbd_pt)
            gamma_values.append(gamma_calc)
            print(f"    Point ({vbd_pt}, {ramp_pt}): gamma = {gamma_calc:.2f}")
        
        # Average gamma across all points
        gamma_avg = np.mean(gamma_values)
        
        # Calculate R-squared for fit quality
        y_pred = slope * x + const
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"  Average gamma: {gamma_avg:.2f}")
        print(f"  R-squared: {r_squared:.4f}")
        
        return gamma_avg, gamma_values
    
    # Calculate gamma for each temperature
    print("="*60)
    print("GAMMA CALCULATION FROM ALL DATA POINTS")
    print("="*60)
    
    gamma_100C, gamma_vals_100C = calculate_gamma_from_data(vbd_ramp_rate_100C, "100°C")
    gamma_150C, gamma_vals_150C = calculate_gamma_from_data(vbd_ramp_rate_150C, "150°C")
    gamma_175C, gamma_vals_175C = calculate_gamma_from_data(vbd_ramp_rate_175C, "175°C")
    gamma_200C, gamma_vals_200C = calculate_gamma_from_data(vbd_ramp_rate_200C, "200°C")
    
    print("\n" + "="*60)
    print("SUMMARY OF CALCULATED GAMMA VALUES")
    print("="*60)
    print(f"100°C  - Calculated gamma: {gamma_100C:.2f} (Paper reported: 11.5)")
    print(f"150°C  - Calculated gamma: {gamma_150C:.2f} (Paper reported: 12.3)")
    print(f"175°C  - Calculated gamma: {gamma_175C:.2f} (Paper reported: 12.4)")
    print(f"200°C  - Calculated gamma: {gamma_200C:.2f} (Paper reported: 9.0)")
    print("="*60 + "\n")

    def plot_gamma_fits():
        """
        Recreate the gamma fits from paper figure 5
        Shows voltage ramp rate vs breakdown voltage at different temperatures
        Uses data defined above: vbd_ramp_rate_100C, vbd_ramp_rate_150C, etc.
        """
        datasets = [
            {
                'data': vbd_ramp_rate_100C,
                'temp': '100 °C',
                'gamma': 11.5,
                'label': '(a)'
            },
            {
                'data': vbd_ramp_rate_150C,
                'temp': '150 °C',
                'gamma': 12.3,
                'label': '(b)'
            },
            {
                'data': vbd_ramp_rate_175C,
                'temp': '175 °C',
                'gamma': 12.4,
                'label': '(c)'
            },
            {
                'data': vbd_ramp_rate_200C,
                'temp': '200 °C',
                'gamma': 9.0,
                'label': '(d)'
            }
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, dataset in enumerate(axes):
            ax = axes[idx]
            
            # Extract vbd and ramp_rate from data
            vbd = np.array([pt[0] for pt in datasets[idx]['data']])
            ramp_rate = np.array([pt[1] for pt in datasets[idx]['data']])
            temp_label = datasets[idx]['temp']
            gamma_nominal = datasets[idx]['gamma']
            subplot_label = datasets[idx]['label']
            
            # Plot data points
            ax.scatter(vbd, ramp_rate, color='red', s=100, zorder=5, marker='o')
            
            # Fit line: ln(ramp_rate) = (gamma - 1) * ln(vbd) + const
            x = np.log(vbd)
            y = np.log(ramp_rate)
            coeffs = np.polyfit(x, y, 1)
            slope, intercept = coeffs[0], coeffs[1]
            
            # Generate fitted line
            vbd_range = np.linspace(50, 200, 200)
            y_fit = slope * np.log(vbd_range) + intercept
            ramp_fit = np.exp(y_fit)
            
            # Plot fitted line
            ax.plot(vbd_range, ramp_fit, 'r-', linewidth=2.5, zorder=3)
            
            # Set scale and limits
            ax.set_yscale('log')
            ax.set_xlim(50, 200)
            ax.set_ylim(1e-2, 1e1)
            
            # Labels
            ax.set_xlabel('Breakdown voltage(V)', fontsize=11)
            ax.set_ylabel('Ramp rate(V/s)', fontsize=11)
            
            # Add subplot label in corner
            ax.text(0.55, 0.05, subplot_label, 
                    transform=ax.transAxes, fontsize=14, fontweight='bold',
                    verticalalignment='bottom', horizontalalignment='right')
            
            # Add gamma label
            ax.text(0.95, 0.95, f'Gamma = {gamma_nominal}', 
                    transform=ax.transAxes, fontsize=12, verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Add temperature label
            ax.text(0.05, 0.90, temp_label, 
                    transform=ax.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)

        fig.suptitle('Voltage Ramp Rate vs Breakdown Voltage (Fig 5)', 
                     fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        filename = 'dielectric_gamma_fits_paper_fig5.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Figure saved as '{filename}'")
        plt.show()
    
    # Call the plotting function
    plot_gamma_fits()


# Call the main function
if __name__ == "__main__":
    hybrid_bonding_dielectric()