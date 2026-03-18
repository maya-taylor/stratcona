"""
Recreate voltage ramp rate vs breakdown voltage plots from paper Fig 5
Shows the gamma field acceleration factor at different temperatures
Source: https://ieeexplore.ieee.org/abstract/document/9764478
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import dielectric module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import data from dielectric module
from demos.hybrid_bonding.dielectric import hybrid_bonding_dielectric

def plot_gamma_fits():
    """
    Recreate the gamma fits from paper figure 5
    Shows voltage ramp rate vs breakdown voltage at different temperatures
    Source: https://ieeexplore.ieee.org/abstract/document/9764478
    """
    # Data from dielectric.py at each temperature
    vbd_ramp_rate_100C = [[150, 0.1], [200, 2.0505]]
    vbd_ramp_rate_150C = [[150, 0.1], [200, 2.5811]]
    vbd_ramp_rate_175C = [[150, 0.1], [200, 2.6564]]
    vbd_ramp_rate_200C = [[150, 0.1], [200, 0.9989]]
    
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
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        # Extract vbd and ramp_rate from data
        vbd = np.array([pt[0] for pt in dataset['data']])
        ramp_rate = np.array([pt[1] for pt in dataset['data']])
        temp_label = dataset['temp']
        gamma_nominal = dataset['gamma']
        subplot_label = dataset['label']
        
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

if __name__ == '__main__':
    plot_gamma_fits()
