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
# I want someway to be able to project cycles to failure but paper I have said they look mostly at gamma

SET_TEMP = 100 # set this to the temperature you want to look at (100, 150, 175, 200)

def hybrid_bonding_dielectric():


    ########################################################
    # Defining Equation
    ########################################################

    # This function calculates log(ramp_rate) from gamma and const
    # Using: ln(ramp_rate) = (gamma - 1) * ln(vbd) + const
    def calc_log_ramprate(vbd, gamma, const):
        log_ramprate = (gamma - 1.0) * jnp.log(vbd) + const
        return log_ramprate

    gamma_prior = {'loc' : 11.5, 'scale' : 2}  # Paper reports gamma = 11.5 at 100C
    const_prior = {'loc' : -60, 'scale' : 20} # fitting constant

    mb = stratcona.SPMBuilder(mdl_name='hb_dielectric')
    mb.add_params(meas_var = 0.2) # measurement variance for log(ramp_rate)
    
    mb.add_hyperlatent('gamma_nom', dists.Normal, gamma_prior)
    mb.add_hyperlatent('const_nom', dists.Normal, const_prior)

    mb.add_latent('gamma', nom='gamma_nom')
    mb.add_latent('const', nom='const_nom')

    print(f"  gamma:     loc={gamma_prior['loc']}, scale={gamma_prior['scale']}")
    print(f"  const:     loc={const_prior['loc']}, scale={const_prior['scale']}")
    print()

    mb.add_intermediate('log_ramprate_predicted', calc_log_ramprate)

    mb.add_observed(
        'log_ramprate_observed',
        dists.Normal,
        {'loc': 'log_ramprate_predicted', 'scale': 'meas_var'},
        1)
    
    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=424242)

    ##################################################################
    # Defining Test Collection and Data

    # Data from paper figure 5

        
    # From 100C, gamma = 11.5 - graph (a)
    vbd_ramp_rate_100C = [[173, 0.05], [195, 0.17]]
    vbd_100C = jnp.array([pt[0] for pt in vbd_ramp_rate_100C])
    ramprate_100C = jnp.array([pt[1] for pt in vbd_ramp_rate_100C])

    # From 150C, gamma = 12.3 - graph (b)
    vbd_ramp_rate_150C = [[148, 0.058], [168, 0.18], [186, 1.2]]
    vbd_150C = jnp.array([pt[0] for pt in vbd_ramp_rate_150C])
    ramprate_150C = jnp.array([pt[1] for pt in vbd_ramp_rate_150C])

    # From 175C, gamma = 12.4 - graph (c)
    vbd_ramp_rate_175C = [[145, 0.052], [152, 0.18], [178, 1.1]]
    vbd_175C = jnp.array([pt[0] for pt in vbd_ramp_rate_175C])
    ramprate_175C = jnp.array([pt[1] for pt in vbd_ramp_rate_175C])


    # From 200C, gamma = 9.0 - graph (d)
    vbd_ramp_rate_200C = [[127, 0.05], [136, 0.18], [168, 1.1]]
    vbd_200C = jnp.array([pt[0] for pt in vbd_ramp_rate_200C])
    ramprate_200C = jnp.array([pt[1] for pt in vbd_ramp_rate_200C])

    #################################################################
    # Define how the data was collected
    #################################################################

    # Create a separate test condition for each ramp_rate value
    # Observe log(ramp_rate), which depends on both gamma and const
    test_conds = {}
    cond_params = {}
    measured_data = {}
    
    if SET_TEMP == 100:
        for i, (ramp_val, vbd_val) in enumerate(zip(ramprate_100C, vbd_100C)):
            test_name = f'temp_100C_test_{i}'
            test_conds[test_name] = {'lot': 1, 'chp': 1}
            cond_params[test_name] = {'vbd': float(vbd_val)}
            # Observe log(ramp_rate) - the model will fit gamma and const to match this
            log_ramprate_obs = float(jnp.log(ramp_val))
            measured_data[test_name] = {'log_ramprate_observed': jnp.array([[[log_ramprate_obs]]])}
    elif SET_TEMP == 150:
        for i, (ramp_val, vbd_val) in enumerate(zip(ramprate_150C, vbd_150C)):
            test_name = f'temp_150C_test_{i}'
            test_conds[test_name] = {'lot': 1, 'chp': 1}
            cond_params[test_name] = {'vbd': float(vbd_val)}
            log_ramprate_obs = float(jnp.log(ramp_val))
            measured_data[test_name] = {'log_ramprate_observed': jnp.array([[[log_ramprate_obs]]])}
    elif SET_TEMP == 175:
        for i, (ramp_val, vbd_val) in enumerate(zip(ramprate_175C, vbd_175C)):
            test_name = f'temp_175C_test_{i}'
            test_conds[test_name] = {'lot': 1, 'chp': 1}
            cond_params[test_name] = {'vbd': float(vbd_val)}
            log_ramprate_obs = float(jnp.log(ramp_val))
            measured_data[test_name] = {'log_ramprate_observed': jnp.array([[[log_ramprate_obs]]])}
    elif SET_TEMP == 200:
        for i, (ramp_val, vbd_val) in enumerate(zip(ramprate_200C, vbd_200C)):
            test_name = f'temp_200C_test_{i}'
            test_conds[test_name] = {'lot': 1, 'chp': 1}
            cond_params[test_name] = {'vbd': float(vbd_val)}
            log_ramprate_obs = float(jnp.log(ramp_val))
            measured_data[test_name] = {'log_ramprate_observed': jnp.array([[[log_ramprate_obs]]])}
    
    accel_test = stratcona.TestDef('accel_test', test_conds, cond_params)
    am.set_test_definition(accel_test)

    #################################################################
    # -------- PRIOR ENTROPY CALCULATION ---------------------------
    #################################################################
    ENTROPY_SAMPLES = 100_000
    
    def lp_f(vals, site, key, test):
        return am.relmdl.logprob(key, test.dims, test.conds, {site: vals}, None, (len(vals),))

    k1, k2 = rand.split(rand.key(9273036857), 2)
    hyl_samples_prior = am.relmdl.sample(k1, accel_test.dims, accel_test.conds, (ENTROPY_SAMPLES,))
    hyls = ['gamma_nom', 'const_nom']
    pri_samples, pri_entropy = {}, {}
    for hyl in hyls:
        pri_samples[hyl] = hyl_samples_prior[hyl]
        pri_entropy[hyl] = stratcona.engine.bed.entropy(
            pri_samples[hyl], partial(lp_f, site=hyl, test=accel_test, key=k1))
    
    print("Prior entropies computed.")

    #################################################################
    # -------- INFERENCE ON THE MODEL ----------------------------
    #################################################################
    start_time = time.time()

    am.do_inference(measured_data)
    print(f'Inference completed in {time.time() - start_time:.2f} seconds')
    print("Posterior hyper-latent beliefs:")
    print(am.relmdl.hyl_beliefs)

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
    # -------- PLOT POSTERIOR DISTRIBUTIONS WITH PRIORS ---------------
    #################################################################
    
    prior_specs = {
        'gamma_nom': gamma_prior,
        'const_nom': const_prior
    }
    
    n_vars = len(pst_samples)
    fig, axes = plt.subplots((n_vars + 1) // 2, 2, figsize=(12, 3 * ((n_vars + 1) // 2)))
    axes = axes.flatten()
    
    for idx, (hyl_name, samples) in enumerate(pst_samples.items()):
        ax = axes[idx]
        
        # Plot posterior histogram
        ax.hist(samples, bins=50, density=True, alpha=0.6, label='Posterior', color='blue', edgecolor='black')
        
        # Plot prior curve
        x_range = jnp.linspace(float(jnp.min(samples)), float(jnp.max(samples)), 200)
        if hyl_name in prior_specs:
            prior_dist = dists.Normal(**prior_specs[hyl_name])
            prior_pdf = jnp.exp(prior_dist.log_prob(x_range))
            ax.plot(x_range, prior_pdf, 'r-', linewidth=2, label='Prior')
        
        ax.set_xlabel(hyl_name)
        ax.set_ylabel('Density')
        ax.set_title(f'{hyl_name}: Prior vs Posterior')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplots
    for idx in range(n_vars, len(axes)):
        fig.delaxes(axes[idx])
    
    fig.suptitle("Prior vs Posterior Distributions", fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    filename = 'dielectric_posterior_distributions.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nDistribution plot saved as '{filename}'")
    plt.show()

    #################################################################
    # -------- PRIOR vs POSTERIOR ENTROPY VIOLIN PLOTS ---------------
    #################################################################
    
    sb.set_context('notebook')
    sb.set_theme(style='ticks', font='Times New Roman')
    
    fig, p = plt.subplots(1, 1, figsize=(10, 6))
    display_map = {
        'gamma_nom': '$\\mu_{\\gamma}$',
        'const_nom': '$\\mu_{const}$'
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
        hue='pri-pst', inner='quart', palette=['skyblue', 'darkblue'], linewidth=1.25
    )
    for fill in p.collections:
        fill.set_alpha(0.75)
    
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman'
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Add text annotations showing entropy and information gain (inside plot, offset above and below)
    pos = range(len(hyls))
    for tick, hyl in enumerate(hyls):
        y_pos = pos[tick]
        # Position entropy labels inside the plot area, stacked vertically offset from center
        p.text(p.get_xlim()[0] + 0.05 * (p.get_xlim()[1] - p.get_xlim()[0]), y_pos - 0.35,
               f'$H_{{prior}}={round(float(pri_entropy[hyl]), 2)}$',
               horizontalalignment='left', size='medium', color='black', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        p.text(p.get_xlim()[0] + 0.05 * (p.get_xlim()[1] - p.get_xlim()[0]), y_pos + 0.35,
               f'$H_{{posterior}}={round(float(pst_entropy[hyl]), 2)}$',
               horizontalalignment='left', size='medium', color='black', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        p.text(p.get_xlim()[1] - 0.05 * (p.get_xlim()[1] - p.get_xlim()[0]), y_pos,
               f'$IG={round(float(hyl_ig[hyl]), 2)}$',
               horizontalalignment='right', verticalalignment='center', size='medium', color='darkgreen', weight='bold', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    p.legend().remove()
    p.tick_params(axis='y', which='major', labelsize=12, labelfontfamily='Times New Roman')
    p.set_xlabel('Parameter Value', fontsize='medium')
    p.set_ylabel('Hyper-latent Variable', fontsize='medium')
    p.set_title('Prior vs Posterior Distributions with Information Gain', fontsize='medium', fontweight='bold')
    
    plt.tight_layout()
    filename_entropy = 'dielectric_prior_posterior_entropy_violin.png'
    plt.savefig(filename_entropy, dpi=150, bbox_inches='tight')
    print(f"\nEntropy violin plot saved as '{filename_entropy}'")
    plt.show()


if __name__ == '__main__':
    hybrid_bonding_dielectric()













