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
ENTROPY_SAMPLES = 100_000


def plot_nuts_trace(chain_samples, parameter_label, filename, max_samples=10000):
    chain_samples = np.asarray(chain_samples)
    single_chain = chain_samples[0, :max_samples]
    draws = np.arange(1, len(single_chain) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(draws, single_chain, linewidth=0.9, alpha=0.9, color="#6d9f60")

    ax.set_xlabel('NUTS Sample', fontsize=12)
    ax.set_ylabel(parameter_label, fontsize=14)
    ax.set_title(f'NUTS Trace for {parameter_label}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"NUTS trace plot saved as '{filename}'")
    plt.show()

def hybrid_bonding_thermomech():
    """
    Fit Engelmaier thermomechanical fatigue model to hybrid bonding data.
    Modify the prior dictionaries below to test different parameter constraints.
    """
    ###################################################################################
    # Fitting Settings
    ###################################################################################
    USE_SCALED_PRIORS = 1
    if USE_SCALED_PRIORS:
        print("Using SCALED priors (c_1 and c_2 scaled up for better numerical stability)")
    

    ##########################################
    # My Fabricated Experimental Data Here
    ###########################################

    # Pairs of points [ % delta_D, Nf (means cycles to failure)]

    # Condition 1: T_sj=50, t_D=600
    delta_D_Nfdata_cond1 = [[0.16198211908340454, 10.0], [0.055673062801361084, 100.0], [0.0191347673535347, 1000.0], [0.006576597224920988, 10000.0], [0.0022603687830269337, 100000.0], [0.0007768860668875277, 1000000.0]]
    delta_D_data_cond1 = jnp.array([pt[0] for pt in delta_D_Nfdata_cond1])
    Nf_data_cond1 = jnp.array([pt[1] for pt in delta_D_Nfdata_cond1])
    
    # Condition 2: T_sj=150, t_D=600
    delta_D_Nfdata_cond2 = delta_D_Nfdata_cond2 = [[0.13533349335193634, 10.0], [0.04051196575164795, 100.0], [0.012127222493290901, 1000.0], [0.003630273975431919, 10000.0], [0.0010867195669561625, 100000.0], [0.0003253085887990892, 1000000.0]]
    delta_D_data_cond2 = jnp.array([pt[0] for pt in delta_D_Nfdata_cond2])
    Nf_data_cond2 = jnp.array([pt[1] for pt in delta_D_Nfdata_cond2])

    # Condition 3: T_sj=40, t_D=300
    delta_D_Nfdata_cond3 = [[0.16768045723438263, 10.0], [0.05918363109230995, 100.0], [0.020889149978756905, 1000.0], [0.007372927851974964, 10000.0], [0.0026023108512163162, 100000.0], [0.0009184982627630234, 1000000.0]]  
    delta_D_data_cond3 = jnp.array([pt[0] for pt in delta_D_Nfdata_cond3])
    Nf_data_cond3 = jnp.array([pt[1] for pt in delta_D_Nfdata_cond3])


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
    # - T_sj: solder joint temperature (Celsius)
    # - delta_D: inelastic (plastic) strain range for cycles (dimensionless ratio delta_L/L)

    #################################################################
    # Define Prior Parameters
    #################################################################
    
    e_f_prior = {'loc': 0.225, 'scale': 0.2}      # fatigue ductility coefficient
    c_0_prior = {'loc': 0.480, 'scale': 0.1}       # base fatigue exponent 
    c_1_prior_default = {'loc': 9.30e-04, 'scale': 1E-03}   # temperature coefficient 
    c_2_prior_default = {'loc': -1.92e-02, 'scale': 1E-01} # dwell time coefficient 
    c_1_prior_scaled = {'loc': 9.30E-04*1E3, 'scale': 1E-03*1E3}   # SCALED: temperature coefficient 
    c_2_prior_scaled = {'loc': -1.92e-02*1E1, 'scale': 1E-01*1E1}  # SCALED: dwell time coefficient 

    if USE_SCALED_PRIORS:   
        c_1_prior = c_1_prior_scaled
        c_2_prior = c_2_prior_scaled
    else:
        c_1_prior = c_1_prior_default
        c_2_prior = c_2_prior_default

    print(f"  e_f:     loc={e_f_prior['loc']}, scale={e_f_prior['scale']}")
    print(f"  c_0:     loc={c_0_prior['loc']}, scale={c_0_prior['scale']}")
    print(f"  c_1:     loc={c_1_prior['loc']}, scale={c_1_prior['scale']}")
    print(f"  c_2:     loc={c_2_prior['loc']}, scale={c_2_prior['scale']}")
    print()

    # caution: values are not clipped or restricted!
    def calc_engelmaier(e_f, c_0, c_1, c_2, t_0, T_sj, t_D, delta_D):
        m =  c_0 + c_1*T_sj + c_2*jnp.log(1 + t_0 / t_D) 
        N_f_50 = 0.5*jnp.power(2*e_f/delta_D, 1/m)
        return N_f_50
    
    def calc_engelmaier_scaled(e_f, c_0, c_1, c_2, t_0, T_sj, t_D, delta_D):
        m =  c_0 + c_1*1E-3*T_sj + c_2*1E-1*jnp.log(1 + t_0 / t_D) 
        N_f_50 = 0.5*jnp.power(2*e_f/delta_D, 1/m)
        return N_f_50
    
    # needed to restrict values to prevent inf/nan from causing divergences
    # want to constrain our variables to be in a reasonable range for them
    def calc_log_engelmaier(e_f, c_0, c_1, c_2, t_0, T_sj, t_D, delta_D):
        e_f, delta_D = jnp.maximum(e_f, 0.001), jnp.maximum(delta_D, 1e-8) # basically restricting to positive values (this realistically cannot be neg)
        m = jnp.maximum(c_0 + c_1*T_sj + c_2*jnp.log(1 + t_0/t_D), 0.1) # normally m is around 0.5, should be solidly around there
        # if i clip m at less that 0.1 this starts diverging a lot, i think a 0.1 clip is reasonable for this application
        # if we want to produce reasonable results, I could also just clip each of the components of m
        N_f = 0.5*jnp.power(jnp.maximum(2*e_f/delta_D, 1e-8), 1/m) # this is just to keep it positive
        log_N_f = jnp.log(jnp.maximum(N_f, 1.0)) # prevent negative numbers
        return jnp.where(jnp.isfinite(log_N_f), log_N_f, jnp.log(1e8))
    
        
    # needed to restrict values to prevent inf/nan from causing divergences
    # want to constrain our variables to be in a reasonable range for them
    def calc_log_engelmaier_scaled(e_f, c_0, c_1, c_2, t_0, T_sj, t_D, delta_D):
        e_f, delta_D = jnp.maximum(e_f, 0.001), jnp.maximum(delta_D, 1e-8) # basically restricting to positive values (this realistically cannot be neg)
        m = jnp.maximum(c_0 + c_1*1E-3*T_sj + c_2*1E-1*jnp.log(1 + t_0/t_D), 0.1) # normally m is around 0.5, should be solidly around there
        # if i clip m at less that 0.1 this starts diverging a lot, i think a 0.1 clip is reasonable for this application
        # if we want to produce reasonable results, I could also just clip each of the components of m
        N_f = 0.5*jnp.power(jnp.maximum(2*e_f/delta_D, 1e-8), 1/m) # this is just to keep it positive
        log_N_f = jnp.log(jnp.maximum(N_f, 1.0)) # prevent negative numbers
        return jnp.where(jnp.isfinite(log_N_f), log_N_f, jnp.log(1e8))
    
    mb = stratcona.SPMBuilder(mdl_name='hb_engelmaier')
    mb.add_params(t_0=400, meas_var = 5)  # Measurement variance in log 
    
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
    
    # using a log since the range of Nf data is very large
    if USE_SCALED_PRIORS:
        mb.add_intermediate('log_engelmaier_nf', calc_log_engelmaier_scaled)
    else:
        mb.add_intermediate('log_engelmaier_nf', calc_log_engelmaier)

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

    # Create a separate test condition for each delta_D value in both conditions
    test_conds = {}
    cond_params = {}
    
    # Condition 1: T_sj=50, t_D=600
    for i, delta_D_val in enumerate(delta_D_data_cond1):
        test_name = f'cond1_test_{i}'
        test_conds[test_name] = {'lot': 1, 'chp': 1}
        cond_params[test_name] = {'T_sj': 50, 't_D': 600, 'delta_D': float(delta_D_val)}
    
    # Condition 2: T_sj=150, t_D=600
    for i, delta_D_val in enumerate(delta_D_data_cond2):
        test_name = f'cond2_test_{i}'
        test_conds[test_name] = {'lot': 1, 'chp': 1}
        cond_params[test_name] = {'T_sj': 150, 't_D': 600, 'delta_D': float(delta_D_val)}
    
    # Condition 3: T_sj=40, t_D=300
    for i, delta_D_val in enumerate(delta_D_data_cond3):
        test_name = f'cond3_test_{i}'
        test_conds[test_name] = {'lot': 1, 'chp': 1}
        cond_params[test_name] = {'T_sj': 40, 't_D': 300, 'delta_D': float(delta_D_val)}
    
    accel_test = stratcona.TestDef('accel_test', test_conds, cond_params)
    am.set_test_definition(accel_test)

    #################################################################
    # -------- PRIOR ENTROPY CALCULATION ---------------------------
    #################################################################
    def lp_f(vals, site, key, test):
        return am.relmdl.logprob(key, test.dims, test.conds, {site: vals}, None, (len(vals),))

    k1, k2 = rand.split(rand.key(9273036857), 2)
    hyl_samples_prior = am.relmdl.sample(k1, accel_test.dims, accel_test.conds, (ENTROPY_SAMPLES,))
    hyls = ['e_f_nom', 'c_0_nom', 'c_1_nom', 'c_2_nom']
    pri_samples, pri_entropy = {}, {}
    for hyl in hyls:
        pri_samples[hyl] = hyl_samples_prior[hyl]
        pri_entropy[hyl] = stratcona.engine.bed.entropy(
            pri_samples[hyl], partial(lp_f, site=hyl, test=accel_test, key=k1))
    
    print("Prior entropies computed.")

    #################################################################
    # -------- INFERENCE ON THE MODEL -----------------------------
    #################################################################
    start_time = time.time()

    # Build measured_data with log-transformed Nf (since we use Normal in log-space)
    measured_data = {}
    
    # Condition 1 data
    for i, nf_val in enumerate(Nf_data_cond1):
        test_name = f'cond1_test_{i}'
        measured_data[test_name] = {'nf_delta_D': jnp.array([[[jnp.log(float(nf_val))]]])}
    
    # Condition 2 data
    for i, nf_val in enumerate(Nf_data_cond2):
        test_name = f'cond2_test_{i}'
        measured_data[test_name] = {'nf_delta_D': jnp.array([[[jnp.log(float(nf_val))]]])}
    
    # Condition 3 data
    for i, nf_val in enumerate(Nf_data_cond3):
        test_name = f'cond3_test_{i}'
        measured_data[test_name] = {'nf_delta_D': jnp.array([[[jnp.log(float(nf_val))]]])}

    inference_result = am.do_inference(measured_data, return_details=True)
    print(f'Inference completed in {time.time() - start_time:.2f} seconds')
    print("Posterior hyper-latent beliefs:")
    print(am.relmdl.hyl_beliefs)

    for hyl in hyls:
        trace_samples = np.asarray(inference_result['samples'][hyl])
        stats = inference_result['convergence_stats'][hyl]
        print(
            f"{hyl} NUTS diagnostics: ESS={float(stats['ess']):.2f}, "
            f"split-Rhat={float(stats['srhat']):.4f}"
        )

        trace_filename = f'hybrid_bonding_{hyl}_nuts_trace.png'
        plot_nuts_trace(trace_samples, hyl, trace_filename)

    #################################################################
    # -------- POSTERIOR ENTROPY CALCULATION -------------------------
    #################################################################
    k1, k2 = rand.split(rand.key(9296245908724), 2)
    hyl_samples_posterior = am.relmdl.sample(k1, accel_test.dims, accel_test.conds, (ENTROPY_SAMPLES,))
    pst_samples, pst_entropy = {}, {}
    for hyl in hyls:
        pst_samples[hyl] = hyl_samples_posterior[hyl]
        pst_entropy[hyl] = stratcona.engine.bed.entropy(
            pst_samples[hyl], partial(lp_f, site=hyl, test=accel_test, key=k1))

    hyl_ig = {}
    for hyl in hyls:
        hyl_ig[hyl] = pri_entropy[hyl] - pst_entropy[hyl]
    
    print("Posterior entropies computed and information gain calculated.")

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
    # Recreating the sampling reproduce plot from Engelmaier paper for SnPb and SAC105 for comparison
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
        if USE_SCALED_PRIORS:
            Nf_hb = calc_engelmaier_scaled(
                e_f_samples[i], c_0_samples[i], c_1_samples[i], c_2_samples[i],
                t_0, T_sj, t_D, delta_D_range
            )
        else:
            Nf_hb = calc_engelmaier(
                e_f_samples[i], c_0_samples[i], c_1_samples[i], c_2_samples[i],
                t_0, T_sj, t_D, delta_D_range
            )
        Nf_hb_all.append(Nf_hb)
    
    Nf_hb_all = jnp.stack(Nf_hb_all)
    
    # Replace inf/nan with reasonable values for visualization
    Nf_hb_all = jnp.where(jnp.isfinite(Nf_hb_all), Nf_hb_all, 1e10)
    # want to keep track of how many inf/nan values needed to be replaced to make sure sampling is ok
    n_inf_replaced = jnp.sum(~jnp.isfinite(Nf_hb_all))
    print(f"Number of inf/nan values replaced: {n_inf_replaced}")
    
    # Compute summary statistics
    def summarize(samples):
        mean = jnp.mean(samples, axis=0)
        lower = jnp.percentile(samples, 2.5, axis=0)
        upper = jnp.percentile(samples, 97.5, axis=0)
        return mean, lower, upper
    
    mean_hb, low_hb, high_hb = summarize(Nf_hb_all)
    
    #################################################################
    # -------- COMPUTE MEAN PREDICTION FROM POSTERIOR MEANS ----------
    #################################################################
    
    # Extract posterior means (loc values) from the fitted distributions
    e_f_mean = float(am.relmdl.hyl_beliefs['e_f_nom']['loc'])
    c_0_mean = float(am.relmdl.hyl_beliefs['c_0_nom']['loc'])
    c_1_mean = float(am.relmdl.hyl_beliefs['c_1_nom']['loc'])
    c_2_mean = float(am.relmdl.hyl_beliefs['c_2_nom']['loc'])
    
    # Calculate deterministic prediction using posterior means
    # recall calc_engelmaier currently does not clip inf values
    if USE_SCALED_PRIORS:
        mean_pred_hb = calc_engelmaier_scaled(
            e_f_mean, c_0_mean, c_1_mean, c_2_mean,
            t_0, T_sj, t_D, delta_D_range
        )
    else:
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
    
    # ---- Posterior Mean ----
    plt.plot(
        mean_pred_hb,
        100*delta_D_range,
        color="#D36CD3",  # purple
        linewidth=3,
        label="Posterior Mean"
    )
    
    # ---- Posterior CI ----
    plt.plot(
        mean_hb,
        100*delta_D_range,
        color="#D36CD3",  # purple
        linewidth=1.5,
        linestyle=":",
        label="Posterior 95% CI",
        alpha=0.7
    )
    
    plt.fill_betweenx(
        100*delta_D_range,
        low_hb,
        high_hb,
        color="#D36CD3",  # purple
        alpha=0.2
    )
    
    plt.plot(low_hb, 100*delta_D_range, color="#D36CD3", linestyle="--", linewidth=1, alpha=0.6)
    plt.plot(high_hb, 100*delta_D_range, color="#D36CD3", linestyle="--", linewidth=1, alpha=0.6)
    
    # ---- SnPb Mean (reference) ----
    # these are the values that my posterior should find
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
        color="#5194F1",  # blue
        linewidth=2.5,
        label="Test Data Mean"
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
        color="#66805f",
        linewidth=2.5,
        label="Prior Mean"
    )
    
    # ---- Data Points (all conditions) ----
    plt.scatter(Nf_data_cond1, 100*delta_D_data_cond1, color="#AD9610", s=100, marker="o", label="Test Data Points", zorder=5)
    # plt.scatter(Nf_data_cond2, 100*delta_D_data_cond2, color="orange", s=100, marker="s", label="Data Cond2 (T=60,t=500)", zorder=5)
    # plt.scatter(Nf_data_cond3, 100*delta_D_data_cond3, color="brown", s=100, marker="^", label="Data Cond3 (T=40,t=300)", zorder=5)
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(10**1, 10**6)
    
    plt.xlabel("Mean Cycles to Failure (Nf, 50%)", fontsize=12)
    plt.ylabel("Inelastic Strain Range (ΔD, %)", fontsize=12)
    #plt.title("Posterior Mean and CI vs. Prior and Test Data Means", fontsize=14, fontweight='bold')
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
    # for now, this isn't really needed since I am just graphing the mean that should be stable
    # but I might want to use the prior vs posterior variance so this is set-up
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

    
    for i in range(n_ref_samples):
        p_snpb = sample_params(keys[i], snpb_nom, snpb_sigma)
        p_sac105 = sample_params(keys[i], sac105_nom, sac105_sigma)
        
        Nf_snpb = calc_engelmaier(
            p_snpb["e_f"], p_snpb["c_0"], p_snpb["c1"], p_snpb["c2"],
            p_snpb["t_0"], T_sj, t_D, delta_D_range
        )
        
        Nf_sac105 = calc_engelmaier(
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
    
    # Use the same prior parameters defined earlier
    prior_specs = {
        'e_f_nom': e_f_prior,
        'c_0_nom': c_0_prior,
        'c_1_nom': c_1_prior,
        'c_2_nom': c_2_prior
    }
    
    # Expected nominal values from SnPb solder data
    if USE_SCALED_PRIORS:
        nominal_values = {
            'e_f_nom': 0.325,
            'c_0_nom': 0.442,
            'c_1_nom': 6.00e-04*1E3,
            'c_2_nom': -1.74e-02*1E1
        }
    else:
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
        if USE_SCALED_PRIORS and hyl_name in ['c_1_nom', 'c_2_nom']:
            ax.set_xlabel(f"{hyl_name}")
        ax.set_title(f'{hyl_name}: Prior vs Posterior')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplots
    for idx in range(n_vars, len(axes)):
        fig.delaxes(axes[idx])
    
    fig.suptitle("Prior vs Posterior Distributions", fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    filename = 'posterior_distributions.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as '{filename}'")
    plt.show()

    #################################################################
    # -------- PRIOR vs POSTERIOR ENTROPY VIOLIN PLOTS ---------------
    #################################################################
    
    sb.set_context('notebook')
    #sb.set_theme(style='ticks', font='Times New Roman')
    
    fig, p = plt.subplots(1, 1, figsize=(10, 6))
    display_map = {
        'e_f_nom': '$\\mu_{e_f}$',
        'c_0_nom': '$\\mu_{c_0}$',
        'c_1_nom': '$\\mu_{c_1}$',
        'c_2_nom': '$\\mu_{c_2}$'
    }
    
    # Create DataFrame combining prior and posterior samples
    df_list = []
    for hyl in hyls:
        hyl_df = pd.DataFrame(pri_samples[hyl], columns=['val'])
        hyl_df['hyl'] = display_map[hyl]
        hyl_df['pri-pst'] = 'Prior'
        df_list.append(hyl_df)
    for hyl in hyls:
        hyl_df = pd.DataFrame(pst_samples[hyl], columns=['val'])
        hyl_df['hyl'] = display_map[hyl]
        hyl_df['pri-pst'] = 'Posterior'
        df_list.append(hyl_df)
    df_violin = pd.concat(df_list, ignore_index=True)
    
    # Create violin plot
    sb.violinplot(
        df_violin, x='val', y='hyl', ax=p, split=True, density_norm='count',
        hue='pri-pst', inner='quart', palette=['#eba5d7', '#83a2cb'], linewidth=1.25
    )
    for fill in p.collections:
        fill.set_alpha(0.75)
    
    # Add text annotations showing entropy and information gain (inside plot, offset above and below)
    pos = range(len(hyls))
    for tick, hyl in enumerate(hyls):
        y_pos = pos[tick]
        # Position entropy labels inside the plot area, stacked vertically offset from center
        p.text(p.get_xlim()[0] + 0.05 * (p.get_xlim()[1] - p.get_xlim()[0]), y_pos - 0.32,
               f'$H_{{prior}}={round(float(pri_entropy[hyl]), 2)}$',
             horizontalalignment='left', fontsize=12, fontweight='semibold', color='black',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=0.8, alpha=0.95))
        p.text(p.get_xlim()[0] + 0.05 * (p.get_xlim()[1] - p.get_xlim()[0]), y_pos + 0.35,
               f'$H_{{posterior}}={round(float(pst_entropy[hyl]), 2)}$',
             horizontalalignment='left', fontsize=12, fontweight='semibold', color='black',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=0.8, alpha=0.95))
        p.text(p.get_xlim()[1] - 0.05 * (p.get_xlim()[1] - p.get_xlim()[0]), y_pos,
               f'$IG={round(float(hyl_ig[hyl]), 2)}$',
             horizontalalignment='right', verticalalignment='center', fontsize=12, color='black', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.45', facecolor='white', edgecolor='black', linewidth=0.9, alpha=0.95))
    
    p.legend().remove()
    p.tick_params(axis='y', which='major', labelsize=12)
    p.set_xlabel('Value Distribution', fontsize='12')
    p.set_ylabel('Latent Variable', fontsize='12')

    plt.tight_layout()
    filename_entropy = 'prior_posterior_entropy_violin.png'
    plt.savefig(filename_entropy, dpi=150, bbox_inches='tight')
    print(f"\nEntropy violin plot saved as '{filename_entropy}'")
    plt.show()


if __name__ == '__main__':
    hybrid_bonding_thermomech()

