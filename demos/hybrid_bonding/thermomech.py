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

import seaborn as sb
from matplotlib import pyplot as plt, scale
import matplotlib.lines as pltlines

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import stratcona

# Review paper suggested that thermomechanical strain is potenial failure mechanism for Cu-Cu hybrid bonding

# Plan for this code --> just use parameters from one of the types of solder to set-up a model
# Then try to figure out parameters that work for hybrid bonding set-up... later problem!


def hybrid_bonding_thermomech():
    """
    Fit Engelmaier thermomechanical fatigue model to hybrid bonding data.
    Modify the prior dictionaries below to test different parameter constraints.
    """
    ##########################################
    # My Fabricated Experimental Data Here
    ###########################################

    # Pairs of points [ % delta_D, Nf (means cycles to failure)]
    # Tried to read of figure 10 from the paper but was guesstimating points
    # delta_D_Nfdata = [[11, 10], [9, 10E2], [1.8,10E3], [0.6,10E4], [0.2,10E5], [0.07, 10E6]] #this was bad data

    # Using data read of the SnPb plot I generated
    delta_D_Nfdata = [[0.16263, 10], [0.0581, 10E2], [0.01916,10E3], [0.00657,10E4], [0.00228,10E5], [0.00078, 10E6]]

    # Convert fabricated data to arrays
    delta_D_data = jnp.array([pt[0] for pt in delta_D_Nfdata])
    Nf_data      = jnp.array([pt[1] for pt in delta_D_Nfdata])

    Nf_avg = jnp.mean(Nf_data)

    ########################################################
    # Defining the Model
    ########################################################

    # Source #1 Solder Creep-Fatigue Model Parameters for SAC & SnAg Lead-Free Solder Joint
    # https://www.circuitinsight.com/pdf/solder_creep_fatigue_ipc.pdf
    # Reliability Estimation by William Engelmaier and Associates

    # Parameter for this equation:
    # - N_f_50: the number to cycles to failure at 50% probability (median fatigue life)
    # - e_f: fatigue ductility coefficient (solder material property)
    # - c_0, c_1, c_2 : empirical found material coefficients
    # - t_0: dwell time at high temperature (seconds)
    # - t_D: reference time constant (seconds)
    # - delta_D: inelastic (plastic) strange range for cycles (dimensionless ratio delta_L/L)

    #################################################################
    # Define Prior Parameters
    #################################################################

    # Using priors from SAC105 and Data from SnPb to see if model will update beliefs accordingly
    e_f_prior = {'loc': 0.225, 'scale': 0.1}      # fatigue ductility coefficient
    c_0_prior = {'loc': 0.480, 'scale': 0.1}  # base fatigue exponent
    c_1_prior = {'loc': 9.30E-04, 'scale': 1E-03}   # temperature coefficient
    c_2_prior = {'loc': -1.92E-02, 'scale': 5E-02}  # dwell time coefficient
    log_sigma_prior = {'loc': 0.5, 'scale': 1}    # log-space measurement variance
    
    print("="*70)
    print("Priors being used:")
    print(f"  e_f:     loc={e_f_prior['loc']}, scale={e_f_prior['scale']}")
    print(f"  c_0:     loc={c_0_prior['loc']}, scale={c_0_prior['scale']}")
    print(f"  c_1:     loc={c_1_prior['loc']}, scale={c_1_prior['scale']}")
    print(f"  c_2:     loc={c_2_prior['loc']}, scale={c_2_prior['scale']}")
    print("="*70)
    print()

    def calc_engelmaier(e_f, c_0, c_1, c_2, t_0, T_sj, t_D, delta_D):
        m =  c_0 + c_1*T_sj + c_2*jnp.log(1 + t_0 / t_D) 
        N_f_50 = 0.5*(2*e_f/delta_D)**(1/m)
        return N_f_50
    
    def calc_log_engelmaier(e_f, c_0, c_1, c_2, t_0, T_sj, t_D, delta_D):
        """Return log of predicted Nf for LogNormal likelihood"""
        N_f = calc_engelmaier(e_f, c_0, c_1, c_2, t_0, T_sj, t_D, delta_D)
        return jnp.log(N_f)
    
    mb = stratcona.SPMBuilder(mdl_name='hb_engelmaier')
    mb.add_params(t_0=400)
    
    mb.add_hyperlatent('e_f_nom', dists.Normal, e_f_prior)
    mb.add_hyperlatent('c_0_nom', dists.Normal, c_0_prior)
    mb.add_hyperlatent('c_1_nom', dists.Normal, c_1_prior)
    mb.add_hyperlatent('c_2_nom', dists.Normal, c_2_prior)
    mb.add_hyperlatent('log_sigma_nom', dists.Normal, log_sigma_prior)
    
    mb.add_latent('e_f', nom='e_f_nom')
    mb.add_latent('c_0', nom='c_0_nom')
    mb.add_latent('c_1', nom='c_1_nom')
    mb.add_latent('c_2', nom='c_2_nom')
    mb.add_latent('log_sigma', nom='log_sigma_nom')
    
    mb.add_intermediate('log_engelmaier_nf', calc_log_engelmaier)

    mb.add_observed(
        'nf_delta_D',
        dists.LogNormal,
        {'loc': 'log_engelmaier_nf', 'scale': 'log_sigma'},
        1  # One observation per test condition
    )

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=424242)

    #################################################################
    # Define how the data was collected
    #################################################################

    # Create a separate test condition for each delta_D value
    test_conds = {}
    for i, delta_D_val in enumerate(delta_D_data):
        test_name = f'test_{i}'
        test_conds[test_name] = {'lot': 1, 'chp': 1}
    
    cond_params = {}
    for i, delta_D_val in enumerate(delta_D_data):
        test_name = f'test_{i}'
        cond_params[test_name] = {'T_sj': 50, 't_D': 600, 'delta_D': float(delta_D_val)}
    
    accel_test = stratcona.TestDef('accel_test', test_conds, cond_params)
    am.set_test_definition(accel_test)

    #################################################################
    # -------- INFERENCE ON THE MODEL -----------------------------
    #################################################################
    start_time = time.time()

    # Build measured_data with one entry per test point
    measured_data = {}
    for i, nf_val in enumerate(Nf_data):
        test_name = f'test_{i}'
        measured_data[test_name] = {'nf_delta_D': jnp.array([nf_val])}

    am.do_inference(measured_data)
    print(f'Inference completed in {time.time() - start_time:.2f} seconds')
    print("Posterior hyper-latent beliefs:")
    print(am.relmdl.hyl_beliefs)

    #################################################################
    # -------- SAMPLE FROM POSTERIOR DISTRIBUTIONS -------------------
    #################################################################
    
    rng_key = rand.key(999)
    posterior_samples = {}
    
    for hyl_name, hyl_params in am.relmdl.hyl_beliefs.items():
        # Create distribution from fitted parameters
        dist = dists.Normal(**hyl_params)
        rng_key, subkey = rand.split(rng_key)
        posterior_samples[hyl_name] = dist.sample(subkey, sample_shape=(5000,))
        print(f"{hyl_name}: mean={float(jnp.mean(posterior_samples[hyl_name])):.6f}, std={float(jnp.std(posterior_samples[hyl_name])):.6f}")

    #################################################################
    # -------- PLOT POSTERIOR DISTRIBUTIONS WITH PRIORS ---------------
    #################################################################
    
    # Use the same prior parameters defined earlier for consistency
    prior_specs = {
        'e_f_nom': e_f_prior,
        'c_0_nom': c_0_prior,
        'c_1_nom': c_1_prior,
        'c_2_nom': c_2_prior,
        'log_sigma_nom': log_sigma_prior
    }
    
    # Expected nominal values from SnPb solder data
    nominal_values = {
        'e_f_nom': 0.325,
        'c_0_nom': 0.442,
        'c_1_nom': 6.00e-04,
        'c_2_nom': -1.74e-02,
    }

    model_mode = "Hybrid Bonding Thermomechanical Model"

    n_vars = len(posterior_samples)
    fig, axes = plt.subplots((n_vars + 1) // 2, 2, figsize=(12, 3 * ((n_vars + 1) // 2)))
    axes = axes.flatten()
    
    for idx, (hyl_name, samples) in enumerate(posterior_samples.items()):
        ax = axes[idx]
        
        # Plot posterior histogram
        ax.hist(samples, bins=50, density=True, alpha=0.6, label='Posterior', color='blue', edgecolor='black')
        
        # Plot prior curve
        x_range = jnp.linspace(float(jnp.min(samples)), float(jnp.max(samples)), 200)
        if hyl_name in prior_specs:
            prior_dist = dists.Normal(**prior_specs[hyl_name])
            prior_pdf = jnp.exp(prior_dist.log_prob(x_range))
            ax.plot(x_range, prior_pdf, 'r-', linewidth=2, label='Prior')
        
        # Plot nominal value line
        if hyl_name in nominal_values:
            nom_val = nominal_values[hyl_name]
            ax.axvline(nom_val, color='green', linestyle='--', linewidth=2.5, label=f'Nominal (SnPb): {nom_val:.4g}')
        
        ax.set_xlabel(hyl_name)
        ax.set_ylabel('Density')
        ax.set_title(f'{hyl_name}: Prior vs Posterior')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplots
    for idx in range(n_vars, len(axes)):
        fig.delaxes(axes[idx])
    
    fig.suptitle(model_mode, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    filename = 'posterior_distributions.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as '{filename}'")
    plt.show()


if __name__ == '__main__':
    hybrid_bonding_thermomech()

