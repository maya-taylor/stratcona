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
from matplotlib import pyplot as plt
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
    ##########################################
    # My Fabricated Experimental Data Here ?
    ###########################################

    # Pairs of points [ % delta_D, Nf (means cycles to failure)]
    delta_D_Nfdata = [[11, 10], [9, 10E2], [1.8,10E3], [0.6,10E4], [0.2,10E5], [0.07, 10E6]]

    # Convert fabricated data to arrays
    delta_D_data = jnp.array([pt[0] for pt in delta_D_Nfdata])/100
    Nf_data      = jnp.array([pt[1] for pt in delta_D_Nfdata])

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

    def calc_engelmaier(e_f, c_0, c_1, c_2, t_0, T_sj, t_D, delta_D):
        m =  c_0 + c_1*T_sj + c_2*jnp.log(1 + t_0 / t_D) 
        N_f_50 = 0.5*(2*e_f/delta_D)**(1/m)
        return N_f_50
    
    mb = stratcona.SPMBuilder(mdl_name='hb_engelmaier')
    mb.add_params(t_0=360)
    
    mb.add_hyperlatent('e_f_nom', dists.Normal, {'loc': 0.325, 'scale': 0.001})
    mb.add_hyperlatent('c_0_nom', dists.Normal, {'loc': 0.442, 'scale': 0.01})
    mb.add_hyperlatent('c_1_nom', dists.Normal, {'loc': 6.00E-04, 'scale': 1E-5})
    mb.add_hyperlatent('c_2_nom', dists.Normal, {'loc': -1.72E-02, 'scale': 5E-04})
    mb.add_hyperlatent('sigma_nom', dists.HalfNormal, {'scale': 0.5})
    
    mb.add_latent('e_f', nom='e_f_nom')
    mb.add_latent('c_0', nom='c_0_nom')
    mb.add_latent('c_1', nom='c_1_nom')
    mb.add_latent('c_2', nom='c_2_nom')
    mb.add_latent('sigma', nom='sigma_nom')
    
    mb.add_intermediate('engelmaier_nf', calc_engelmaier)

    mb.add_observed(
        'nf_delta_D',
        dists.Normal,
        {'loc': 'engelmaier_nf', 'scale': 'sigma'},
        len(delta_D_data)
    )

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=424242)

    #################################################################
    # Define how the data was collected
    #################################################################

    # These are the numbers taken from Fig 10 in the paper
    accel_test = stratcona.TestDef(
        'accel_test',
        {'at_50C_10m': {'lot': 1, 'chp': len(delta_D_data)}},
        {'at_50C_10m': {'T_sj': 50, 't_D': 600, 'delta_D': delta_D_data}}
    )
    am.set_test_definition(accel_test)

    #################################################################
    # -------- INFERENCE ON THE MODEL -----------------------------
    #################################################################
    start_time = time.time()

    measured_data = {
        'at_50C_10m': {            
            'nf_delta_D': Nf_data  
        }
    }

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
    # -------- PLOT POSTERIOR DISTRIBUTIONS --------------------------
    #################################################################
    
    n_vars = len(posterior_samples)
    fig, axes = plt.subplots((n_vars + 1) // 2, 2, figsize=(12, 3 * ((n_vars + 1) // 2)))
    axes = axes.flatten()
    
    for idx, (hyl_name, samples) in enumerate(posterior_samples.items()):
        ax = axes[idx]
        ax.hist(samples, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.set_xlabel(hyl_name)
        ax.set_ylabel('Density')
        ax.set_title(f'Posterior: {hyl_name}')
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplots
    for idx in range(n_vars, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('posterior_distributions.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'posterior_distributions.png'")
    plt.show()

hybrid_bonding_thermomech()

