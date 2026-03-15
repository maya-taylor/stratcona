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
    
    e_f_prior = {'loc': 0.225, 'scale': 0.2}      # fatigue ductility coefficient
    c_0_prior = {'loc': 0.480, 'scale': 0.1}       # base fatigue exponent 
    c_1_prior = {'loc': 9.30e-04, 'scale': 3E-04}   # temperature coefficient 
    c_2_prior = {'loc': -1.92e-02, 'scale': 3E-03} # dwell time coefficient 


    print(f"  e_f:     loc={e_f_prior['loc']}, scale={e_f_prior['scale']}")
    print(f"  c_0:     loc={c_0_prior['loc']}, scale={c_0_prior['scale']}")
    print(f"  c_1:     loc={c_1_prior['loc']}, scale={c_1_prior['scale']}")
    print(f"  c_2:     loc={c_2_prior['loc']}, scale={c_2_prior['scale']}")
    print()

    def calc_engelmaier(e_f, c_0, c_1, c_2, t_0, T_sj, t_D, delta_D):
        m =  c_0 + c_1*T_sj + c_2*jnp.log(1 + t_0 / t_D) 
        N_f_50 = 0.5*jnp.power(2*e_f/delta_D, 1/m)
        return N_f_50
    
    def calc_log_engelmaier(e_f, c_0, c_1, c_2, t_0, T_sj, t_D, delta_D):
        e_f, delta_D = jnp.maximum(e_f, 0.001), jnp.maximum(delta_D, 1e-8)
        m = jnp.maximum(c_0 + c_1*T_sj + c_2*jnp.log(1 + t_0/t_D), 0.1)
        N_f = 0.5*jnp.power(jnp.maximum(2*e_f/delta_D, 1e-8), 1/m)
        log_N_f = jnp.log(jnp.maximum(N_f, 1.0))
        return jnp.where(jnp.isfinite(log_N_f), log_N_f, jnp.log(1e8))
    
    mb = stratcona.SPMBuilder(mdl_name='hb_engelmaier')
    mb.add_params(t_0=400, meas_var = 7)  # Measurement variance in log 
    
    mb.add_hyperlatent('e_f_nom', dists.Normal, e_f_prior)
    mb.add_hyperlatent('c_0_nom', dists.Normal, c_0_prior)
    mb.add_hyperlatent('c_1_nom', dists.Normal, c_1_prior)
    mb.add_hyperlatent('c_2_nom', dists.Normal, c_2_prior)
    
    # Ian mentioned defining latents below w/o adding variance is redundant
    # need to fix references 
    mb.add_latent('e_f', nom='e_f_nom') 
    mb.add_latent('c_0', nom='c_0_nom')
    mb.add_latent('c_1', nom='c_1_nom')
    mb.add_latent('c_2', nom='c_2_nom')
    
    mb.add_intermediate('log_engelmaier_nf', calc_log_engelmaier)
    #mb.add_intermediate('engelmaier_nf', calc_engelmaier)


    mb.add_observed(
        'nf_delta_D',
        dists.Normal,  # Normal dist in log-space (not LogNormal)
        {'loc': 'log_engelmaier_nf', 'scale': 'meas_var'},
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

    # Build measured_data with log-transformed Nf (since we use Normal in log-space)
    measured_data = {}
    for i, nf_val in enumerate(Nf_data):
        test_name = f'test_{i}'
        measured_data[test_name] = {'nf_delta_D': jnp.array([jnp.log(float(nf_val))])}

    am.do_inference(measured_data)
    print(f'Inference completed in {time.time() - start_time:.2f} seconds')
    print("Posterior hyper-latent beliefs:")
    print(am.relmdl.hyl_beliefs)

    #################################################################
    # -------- SAMPLE FROM POSTERIOR DISTRIBUTIONS -------------------
    #################################################################
    
    rng_key = rand.key(42424242)
    posterior_samples = {}
    
    for hyl_name, hyl_params in am.relmdl.hyl_beliefs.items():
        # Create distribution from fitted parameters
        dist = dists.Normal(**hyl_params)
        rng_key, subkey = rand.split(rng_key)
        posterior_samples[hyl_name] = dist.sample(subkey, sample_shape=(5000,))
        print(f"{hyl_name}: mean={float(jnp.mean(posterior_samples[hyl_name])):.6f}, std={float(jnp.std(posterior_samples[hyl_name])):.6f}")

    #################################################################
    # -------- GENERATE POSTERIOR PREDICTIONS -------------------------
    #################################################################
    
    # Generate Nf predictions across delta_D range using posterior samples
    delta_D_range = jnp.logspace(-4, 0, 1000)
    T_sj = 50
    t_D = 600
    t_0 = 400
    
    # Extract posterior samples as arrays
    e_f_samples = posterior_samples['e_f_nom']
    c_0_samples = posterior_samples['c_0_nom']
    c_1_samples = posterior_samples['c_1_nom']
    c_2_samples = posterior_samples['c_2_nom']
    
    n_samples_posterior = len(e_f_samples)
    Nf_hb_all = []
    
    for i in range(n_samples_posterior):
        Nf_hb = calc_engelmaier(
            e_f_samples[i], c_0_samples[i], c_1_samples[i], c_2_samples[i],
            t_0, T_sj, t_D, delta_D_range
        )
        Nf_hb_all.append(Nf_hb)
    
    Nf_hb_all = jnp.stack(Nf_hb_all)
    
    # Replace inf/nan with reasonable values for visualization
    Nf_hb_all = jnp.where(jnp.isfinite(Nf_hb_all), Nf_hb_all, 1e10)
    
    # Compute summary statistics
    def summarize(samples):
        mean = jnp.mean(samples, axis=0)
        lower = jnp.percentile(samples, 2.5, axis=0)
        upper = jnp.percentile(samples, 97.5, axis=0)
        return mean, lower, upper
    
    mean_hb, low_hb, high_hb = summarize(Nf_hb_all)
    
    # Replace remaining inf values
    mean_hb = jnp.where(jnp.isfinite(mean_hb), mean_hb, 1e10)
    low_hb = jnp.where(jnp.isfinite(low_hb), low_hb, 1e1)
    high_hb = jnp.where(jnp.isfinite(high_hb), high_hb, 1e10)
    
    #################################################################
    # -------- COMPUTE MEAN PREDICTION FROM POSTERIOR MEANS ----------
    #################################################################
    
    # Extract posterior means (loc values) from the fitted distributions
    e_f_mean = float(am.relmdl.hyl_beliefs['e_f_nom']['loc'])
    c_0_mean = float(am.relmdl.hyl_beliefs['c_0_nom']['loc'])
    c_1_mean = float(am.relmdl.hyl_beliefs['c_1_nom']['loc'])
    c_2_mean = float(am.relmdl.hyl_beliefs['c_2_nom']['loc'])
    
    # Calculate deterministic prediction using posterior means
    mean_pred_hb = calc_engelmaier(
        e_f_mean, c_0_mean, c_1_mean, c_2_mean,
        t_0, T_sj, t_D, delta_D_range
    )
    
    print(f"\nPosterior mean parameters:")
    print(f"  e_f: {e_f_mean:.6f}")
    print(f"  c_0: {c_0_mean:.6f}")
    print(f"  c_1: {c_1_mean:.8f}")
    print(f"  c_2: {c_2_mean:.8f}")
    
    #################################################################
    # -------- PLOT POSTERIOR vs DELTA_D (LIKE SAMPLING REPRODUCE) ----
    #################################################################
    
    plt.figure(figsize=(8, 6))
    
    # ---- Hybrid Bonding (posterior mean from distribution means) ----
    plt.plot(
        mean_pred_hb,
        100*delta_D_range,
        color="purple",
        linewidth=3,
        label="Posterior Mean"
    )
    
    # ---- Hybrid Bonding (posterior ensemble) ----
    plt.plot(
        mean_hb,
        100*delta_D_range,
        color="purple",
        linewidth=1.5,
        linestyle=":",
        label="Posterior 95% CI",
        alpha=0.7
    )
    
    plt.fill_betweenx(
        100*delta_D_range,
        low_hb,
        high_hb,
        color="purple",
        alpha=0.2
    )
    
    plt.plot(low_hb, 100*delta_D_range, color="purple", linestyle="--", linewidth=1, alpha=0.6)
    plt.plot(high_hb, 100*delta_D_range, color="purple", linestyle="--", linewidth=1, alpha=0.6)
    
    # ---- SnPb Mean (reference) ----
    snpb_nom_vals = {
        "e_f": 0.325,
        "c_0": 0.442,
        "c1": 6.00e-04,
        "c2": -1.74e-02,
        "t_0": 360
    }
    
    Nf_snpb_mean = calc_engelmaier(
        snpb_nom_vals["e_f"], snpb_nom_vals["c_0"], snpb_nom_vals["c1"], snpb_nom_vals["c2"],
        snpb_nom_vals["t_0"], T_sj, t_D, delta_D_range
    )
    
    plt.plot(
        Nf_snpb_mean,
        100*delta_D_range,
        color="blue",
        linewidth=2.5,
        label="SnPb Mean"
    )
    
    # ---- SAC105 Mean (reference) ----
    sac105_nom_vals = {
        "e_f": 0.225,
        "c_0": 0.480,
        "c1": 9.30e-04,
        "c2": -1.92e-02,
        "t_0": 500
    }
    
    Nf_sac105_mean = calc_engelmaier(
        sac105_nom_vals["e_f"], sac105_nom_vals["c_0"], sac105_nom_vals["c1"], sac105_nom_vals["c2"],
        sac105_nom_vals["t_0"], T_sj, t_D, delta_D_range
    )
    
    plt.plot(
        Nf_sac105_mean,
        100*delta_D_range,
        color="green",
        linewidth=2.5,
        label="SAC105 Mean"
    )
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(10**1, 10**6)
    
    plt.xlabel("Mean Cycles to Failure (Nf, 50%)", fontsize=12)
    plt.ylabel("Inelastic Strain Range (ΔD, %)", fontsize=12)
    plt.title("Posterior vs SAC105 and SnPb (Posterior should be SnPb)", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    
    filename_posterior = 'hybrid_bonding_posterior.png'
    plt.savefig(filename_posterior, dpi=150, bbox_inches='tight')
    print(f"\nPosterior plot saved as '{filename_posterior}'")
    plt.show()

    #################################################################
    # -------- PLOT POSTERIOR PREDICTIONS WITH REFERENCE SOLDERS ------
    #################################################################
    
    # Reference solder data from Engelmaier Paper
    snpb_nom = {
        "e_f": 0.325,
        "c_0": 0.442,
        "c1": 6.00e-04,
        "c2": -1.74e-02,
        "t_0": 360
    }
    
    snpb_sigma = {
        "e_f": 0.01,
        "c_0": 0.005,
        "c1":  1e-05,
        "c2":  5e-04,
        "t_0": 5.0
    }
    
    sac105_nom = {
        "e_f": 0.225,
        "c_0": 0.480,
        "c1": 9.30e-04,
        "c2": -1.92e-02,
        "t_0": 500
    }
    
    sac105_sigma = {
        "e_f": 0.01,
        "c_0": 0.005,
        "c1":  1e-05,
        "c2":  5e-04,
        "t_0": 5.0
    }
    
    # Generate reference predictions
    def sample_params(key, nom_dict, sigma_dict):
        keys = rand.split(key, len(nom_dict))
        sampled = {}
        for i, k in enumerate(nom_dict):
            mu = nom_dict[k]
            sigma = sigma_dict[k]
            sampled[k] = mu + sigma * rand.normal(keys[i])
        return sampled
    
    rng_key_ref = rand.key(42)
    n_ref_samples = 500
    Nf_snpb_all = []
    Nf_sac105_all = []
    
    keys = rand.split(rng_key_ref, n_ref_samples)
    
    def calc_engelmaier_ref(e_f, c_0, c_1, c_2, t_0, T_sj, t_D, delta_D):
        m = c_0 + c_1*T_sj + c_2*jnp.log(1 + t_0 / t_D)
        N_f_50 = 0.5*(2*e_f/delta_D)**(1/m)
        return N_f_50
    
    for i in range(n_ref_samples):
        p_snpb = sample_params(keys[i], snpb_nom, snpb_sigma)
        p_sac105 = sample_params(keys[i], sac105_nom, sac105_sigma)
        
        Nf_snpb = calc_engelmaier_ref(
            p_snpb["e_f"], p_snpb["c_0"], p_snpb["c1"], p_snpb["c2"],
            p_snpb["t_0"], T_sj, t_D, delta_D_range
        )
        
        Nf_sac105 = calc_engelmaier_ref(
            p_sac105["e_f"], p_sac105["c_0"], p_sac105["c1"], p_sac105["c2"],
            p_sac105["t_0"], T_sj, t_D, delta_D_range
        )
        
        Nf_snpb_all.append(Nf_snpb)
        Nf_sac105_all.append(Nf_sac105)
    
    Nf_snpb_all = jnp.stack(Nf_snpb_all)
    Nf_sac105_all = jnp.stack(Nf_sac105_all)

    
    #################################################################
    # -------- PLOT POSTERIOR DISTRIBUTIONS WITH PRIORS ---------------
    #################################################################
    
    # Use the same prior parameters defined earlier for consistency
    prior_specs = {
        'e_f_nom': e_f_prior,
        'c_0_nom': c_0_prior,
        'c_1_nom': c_1_prior,
        'c_2_nom': c_2_prior
    }
    
    # Expected nominal values from SnPb solder data
    nominal_values = {
        'e_f_nom': 0.325,
        'c_0_nom': 0.442,
        'c_1_nom': 6.00e-04,
        'c_2_nom': -1.74e-02
    }

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
    
    fig.suptitle("Model", fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    filename = 'posterior_distributions.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as '{filename}'")
    plt.show()


if __name__ == '__main__':
    hybrid_bonding_thermomech()

