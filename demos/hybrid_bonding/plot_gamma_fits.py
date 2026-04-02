"""
Recreate voltage ramp rate vs breakdown voltage plots from paper Fig 5
Shows the gamma field acceleration factor at different temperatures
Source: https://ieeexplore.ieee.org/abstract/document/9764478
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
import sys
import os

# Add parent directory to path to import dielectric module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def plot_gamma_fits():
    """
    Recreate the gamma fits from paper figure 5
    Shows voltage ramp rate vs breakdown voltage at different temperatures
    Power law fit: ramp_rate = prefactor * vbd^gamma  (fit directly in original space)
    Source: https://ieeexplore.ieee.org/abstract/document/9764478
    """
    # Data points extracted from paper Fig 5 (same as datagen_dielectric.py)
    vbd_ramp_rate_100C = [[173, 0.05], [195, 0.17]]
    vbd_ramp_rate_150C = [[148, 0.058], [168, 0.18], [186, 1.2]]
    vbd_ramp_rate_175C = [[145, 0.052], [152, 0.18], [178, 1.1]]
    vbd_ramp_rate_200C = [[127, 0.05], [136, 0.18], [168, 1.1]]

    datasets = [
        {'data': vbd_ramp_rate_100C, 'temp': '100 °C', 'gamma_paper': 11.5, 'label': '(a)'},
        {'data': vbd_ramp_rate_150C, 'temp': '150 °C', 'gamma_paper': 12.3, 'label': '(b)'},
        {'data': vbd_ramp_rate_175C, 'temp': '175 °C', 'gamma_paper': 12.4, 'label': '(c)'},
        {'data': vbd_ramp_rate_200C, 'temp': '200 °C', 'gamma_paper': 9.0,  'label': '(d)'},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        vbd      = np.array([pt[0] for pt in dataset['data']])
        ramprate = np.array([pt[1] for pt in dataset['data']])
        temp_label    = dataset['temp']
        gamma_paper   = dataset['gamma_paper']
        subplot_label = dataset['label']

        # Power-law fit directly in original space: ramp_rate = prefactor * vbd^gamma
        def power_law(v, prefactor, gamma):
            return prefactor * v ** gamma

        p0 = [1e-20, 10.0]  # initial guess: prefactor, gamma
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            popt, _ = curve_fit(power_law, vbd, ramprate, p0=p0, maxfev=10000)
        prefactor_fit, gamma_fit = popt

        # Plot data points
        ax.scatter(vbd, ramprate, color='red', s=100, zorder=5, marker='o', label='Data')

        # Plot power-law fit curve over a wider voltage range
        vbd_range = np.linspace(min(vbd) * 0.85, max(vbd) * 1.10, 300)
        ramp_fit  = power_law(vbd_range, prefactor_fit, gamma_fit)
        ax.plot(vbd_range, ramp_fit, 'r-', linewidth=2.5, zorder=3,
                label=f'Power law fit  $\\gamma={gamma_fit:.2f}$')

        ax.set_yscale('log')
        ax.set_xlim(100, 220)
        ax.set_ylim(1e-2, 1e1)

        ax.set_xlabel('Breakdown voltage (V)',  fontsize=11)
        ax.set_ylabel('Ramp rate (V/s)',        fontsize=11)

        ax.text(0.55, 0.05, subplot_label,
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                verticalalignment='bottom', horizontalalignment='right')

        ax.text(0.97, 0.95,
                f'$\\gamma_{{paper}}={gamma_paper}$\n$\\gamma_{{fit}}={gamma_fit:.2f}$',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.text(0.05, 0.90, temp_label,
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

    fig.suptitle('Voltage Ramp Rate vs Breakdown Voltage — Power Law Fit  ($R = c \cdot V_{bd}^{\gamma}$)',
                 fontsize=13, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    filename = 'dielectric_gamma_fits_paper_fig5.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Figure saved as '{filename}'")
    plt.show()

if __name__ == '__main__':
    plot_gamma_fits()
