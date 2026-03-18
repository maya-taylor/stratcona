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

# This script is for modelling the reliability of hybrid bonding interconnects from dielectric degradation
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
    # Each dataset has two points that produce the labeled gamma value when fitted
    
    # From 100C, gamma = 11.5
    vbd_ramp_rate_100C = [[150, 0.1], [200, 2.05]]

    # From 150C, gamma = 12.3
    vbd_ramp_rate_150C = [[150, 0.1], [200, 2.60]]

    # From 175C, gamma = 12.4
    vbd_ramp_rate_175C = [[150, 0.1], [200, 2.65]]

    # From 200C, gamma = 9.0
    vbd_ramp_rate_200C = [[150, 0.1], [200, 1]]

    ########################################################
    # Plot Recreation of Paper Fig 5
    ########################################################
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



