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

def plot_nuts_trace(chain_samples, parameter_label, filename, max_samples=1000, warmup_samples=None):
    chain_samples = np.asarray(chain_samples)
    single_chain = chain_samples[0, :max_samples]

    fig, ax = plt.subplots(figsize=(10, 5))

    if warmup_samples is not None:
        warmup_samples = np.asarray(warmup_samples)
        warmup_chain = warmup_samples[0, :max_samples]
        warmup_draws = np.arange(1, len(warmup_chain) + 1)
        ax.plot(warmup_draws, warmup_chain, linewidth=1.0, alpha=1, color="#c1ae64", label='Warmup samples')

        post_draws = np.arange(len(warmup_chain) + 1, len(warmup_chain) + len(single_chain) + 1)
        ax.plot(post_draws, single_chain, linewidth=1.0, alpha=1, color="#77ae6a", label='Posterior samples')
        ax.set_xlim(1, len(warmup_chain) + len(single_chain))
    else:
        draws = np.arange(1, len(single_chain) + 1)
        ax.plot(draws, single_chain, linewidth=1.0, alpha=0.9, color="#77ae6a", label='Posterior samples')

    ax.set_xlabel('NUTS Sample', fontsize=12)
    ax.set_ylabel(parameter_label, fontsize=14)
  #  ax.set_title(f'NUTS Trace for {parameter_label}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.25)
    if warmup_samples is not None:
        ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"NUTS trace plot saved as '{filename}'")
    plt.show()

BOLTZ_EV = 8.617e-5 #boltzmann constant in eV/K

# This paper has Black's parameters for hybrid bonding
# It is a bit old but I think it will do
# https://ieeexplore.ieee.org/abstract/document/7936378

def hybrid_bonding_electromigration():

    #######################################################
    # Define Equation
    #######################################################

    # note that I may want this as a logarithm instead
    def calc_blacks(current_density, temp, ea, A, k, n):
        eps = 1e-12
        ea_pos = jnp.maximum(ea, eps)
        n_pos = jnp.maximum(n, eps)
        A_pos = jnp.maximum(A, eps)
        mttf = A_pos / pow(current_density, n_pos) * jnp.exp(ea_pos / (k * temp))
        return mttf

    # Defining priors using the paper listed above
    # ea_prior     = {'loc': 0.99, 'scale': 0.02} # in eV, using Boltzmann in eV/K
    # n_prior      = {'loc': 1.36, 'scale': 0.07}
    # A_prior      = {'loc': 0.516, 'scale': 0.1} # hours, calibrated for 350C/20mA with per-via J
    
    ea_prior     = {'loc': 0.99, 'scale': 0.02} # in eV, using Boltzmann in eV/K
    n_prior      = {'loc': 1.36, 'scale': 0.07}
    A_prior      = {'loc': 0.516, 'scale': 0.2} # hours, calibrated for 350C/20mA with per-via J

    # Define the model
    mb = stratcona.SPMBuilder(mdl_name='hb_electromigration')

    # Add fixed stochastic parameters
    mb.add_hyperlatent('ea', dists.Normal, ea_prior)
    mb.add_hyperlatent('n', dists.Normal, n_prior)
    mb.add_hyperlatent('A', dists.Normal, A_prior)

    # Add all parameters + define equation
    current_density = (20e-3 / 16) / (3.6**2 / 1e8)  # A/cm^2, split over 16 vias of 3.6um x 3.6um
    mb.add_params(temp=350 + 273.15, current_density=current_density, k=BOLTZ_EV, meas_var=5)
    mb.add_intermediate('blacks_eq', calc_blacks)

    mb.add_observed(
        'mttf_em',
        dists.Normal,
        {'loc': 'blacks_eq', 'scale': 'meas_var'},
        5
    )

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed = 44242424)
    ##########################################################
    # Define data
    ##########################################################

    mttf_data = [211, 214, 215, 216, 218] # rough points taken from black points in Fig 4
                                         # Variance should be around 5-10 based on graph (meas_var)
    
    #########################################################
    # Set-up How Test Data was Acquired!
    #########################################################
    test_conds = {'em_test': {'lot': 1, 'chp': 1}}
    cond_params = {'em_test': {}}
    measured_data = {'em_test': {'mttf_em': jnp.array([[[float(v) for v in mttf_data]]])}}

    accel_test = stratcona.TestDef('accel_test', test_conds, cond_params)
    am.set_test_definition(accel_test)

    #################################################################
    # -------- PRIOR ENTROPY CALCULATION ---------------------------
    #################################################################
    entropy_samples = 100_000

    def lp_f(vals, site, key, test):
        return am.relmdl.logprob(key, test.dims, test.conds, {site: vals}, None, (len(vals),))

    k1, k2 = rand.split(rand.key(9027341123), 2)
    hyl_samples_prior = am.relmdl.sample(k1, accel_test.dims, accel_test.conds, (entropy_samples,))
    hyls = ['ea', 'n', 'A']
    pri_samples, pri_entropy = {}, {}
    for hyl in hyls:
        pri_samples[hyl] = hyl_samples_prior[hyl]
        pri_entropy[hyl] = stratcona.engine.bed.entropy(
            pri_samples[hyl], partial(lp_f, site=hyl, test=accel_test, key=k1))

    print("Prior entropies computed.")

    #################################################################
    # -------- INFERENCE ON THE MODEL -------------------------------
    #################################################################
    start_time = time.time()
    inference_result = am.do_inference(measured_data, return_details=True, collect_warmup=True)
    print(f'Inference completed in {time.time() - start_time:.2f} seconds')
    print("Posterior hyper-latent beliefs:")
    print(am.relmdl.hyl_beliefs)

    for hyl in hyls:
        trace_samples = np.asarray(inference_result['samples'][hyl])
        warmup_trace_samples = np.asarray(inference_result['warmup_samples'][hyl])
        stats = inference_result['convergence_stats'][hyl]
        trace_filename = f'electromigration_{hyl}_nuts_trace.png'
        plot_nuts_trace(
            trace_samples,
            hyl,
            trace_filename,
            max_samples=2000,
            warmup_samples=warmup_trace_samples,
        )

    #################################################################
    # -------- POSTERIOR ENTROPY CALCULATION ------------------------
    #################################################################
    k1, k2 = rand.split(rand.key(6730912345), 2)
    hyl_samples_posterior = am.relmdl.sample(k1, accel_test.dims, accel_test.conds, (entropy_samples,))
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
    # -------- PLOT POSTERIOR DISTRIBUTIONS WITH PRIORS ------------
    #################################################################
    prior_specs = {'ea': ea_prior, 'n': n_prior, 'A': A_prior}

    n_vars = len(pst_samples)
    fig, axes = plt.subplots((n_vars + 1) // 2, 2, figsize=(12, 3 * ((n_vars + 1) // 2)))
    axes = axes.flatten()

    for idx, (hyl_name, samples) in enumerate(pst_samples.items()):
        ax = axes[idx]
        ax.hist(samples, bins=50, density=True, alpha=0.6, label='Posterior', color='blue', edgecolor='black')

        x_range = jnp.linspace(float(jnp.min(samples)), float(jnp.max(samples)), 200)
        prior_dist = dists.Normal(**prior_specs[hyl_name])
        prior_pdf = jnp.exp(prior_dist.log_prob(x_range))
        ax.plot(x_range, prior_pdf, 'r-', linewidth=2, label='Prior')

        ax.set_xlabel(hyl_name)
        ax.set_ylabel('Density')
        ax.set_title(f'{hyl_name}: Prior vs Posterior')
        ax.legend()
        ax.grid(True, alpha=0.3)

    for idx in range(n_vars, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle("Prior vs Posterior Distributions", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    filename = 'electromigration_posterior_distributions.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nDistribution plot saved as '{filename}'")
    plt.show()

    #################################################################
    # -------- PRIOR vs POSTERIOR ENTROPY VIOLIN PLOTS ------------
    #################################################################
    sb.set_context('notebook')
    sb.set_theme(style='ticks', font='Times New Roman')

    fig, p = plt.subplots(1, 1, figsize=(10, 6))
    display_map = {
        'ea': '$E_a$',
        'n': '$n$',
        'A': '$A$'
    }

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

    pos = range(len(hyls))
    for tick, hyl in enumerate(hyls):
        y_pos = pos[tick]
        p.text(p.get_xlim()[0] + 0.05 * (p.get_xlim()[1] - p.get_xlim()[0]), y_pos - 0.35,
               f'$H_{{prior}}={round(float(pri_entropy[hyl]), 2)}$',
               horizontalalignment='left', size='medium', color='black',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        p.text(p.get_xlim()[0] + 0.05 * (p.get_xlim()[1] - p.get_xlim()[0]), y_pos + 0.35,
               f'$H_{{posterior}}={round(float(pst_entropy[hyl]), 2)}$',
               horizontalalignment='left', size='medium', color='black',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        p.text(p.get_xlim()[1] - 0.05 * (p.get_xlim()[1] - p.get_xlim()[0]), y_pos,
               f'$IG={round(float(hyl_ig[hyl]), 2)}$',
               horizontalalignment='right', verticalalignment='center', size='medium',
               color='darkgreen', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    p.legend().remove()
    p.tick_params(axis='y', which='major', labelsize=12, labelfontfamily='Times New Roman')
    p.set_xlabel('Parameter Value', fontsize='medium')
    p.set_ylabel('Hyper-latent Variable', fontsize='medium')
    p.set_title('Prior vs Posterior Distributions with Information Gain', fontsize='medium', fontweight='bold')

    plt.tight_layout()
    filename_entropy = 'electromigration_prior_posterior_entropy_violin.png'
    plt.savefig(filename_entropy, dpi=150, bbox_inches='tight')
    print(f"\nEntropy violin plot saved as '{filename_entropy}'")
    plt.show()

    #################################################################
    # -------- PLOT COLOR-CODED MTTF PREDICTIONS (PRIOR vs POSTERIOR)
    #################################################################
    num_curves = 1000
    test_plot = stratcona.TestDef('mttf_plot', test_conds, cond_params)
    rng, k1, k2, k3, k4 = rand.split(rand.key(4525455524242), 5)

    # Sample from prior and compute MTTF curves over a range of temperatures
    prm_samples_prior = am.relmdl.sample(k1, test_plot.dims, test_plot.conds, (num_curves,), ('ea', 'n', 'A'))
    pri_sample_probs = jnp.exp(am.relmdl.logprob(k2, test_plot.dims, test_plot.conds, prm_samples_prior, None, (num_curves,)))
    pri_sample_probs = pri_sample_probs / (jnp.max(pri_sample_probs) * 2)

    # Temperature range for MTTF curves (100°C to 400°C in Celsius, converted to Kelvin for calculation)
    temp_range_celsius = jnp.linspace(100, 400, 100)
    temp_range_kelvin = temp_range_celsius + 273.15
    temps_for_plot = jnp.full((num_curves, 100), temp_range_kelvin)  # Shape: (num_curves, 100)

    # Compute prior MTTF predictions
    pri_mttf = calc_blacks(
        current_density,
        temps_for_plot,
        prm_samples_prior['ea'].reshape(-1, 1),
        prm_samples_prior['A'].reshape(-1, 1),
        BOLTZ_EV,
        prm_samples_prior['n'].reshape(-1, 1)
    )  # Shape: (num_curves, 100)

    # Sample from posterior and compute MTTF curves
    prm_samples_posterior = am.relmdl.sample(k3, test_plot.dims, test_plot.conds, (num_curves,), ('ea', 'n', 'A'))
    pst_sample_probs = jnp.exp(am.relmdl.logprob(k4, test_plot.dims, test_plot.conds, prm_samples_posterior, None, (num_curves,)))
    pst_sample_probs = pst_sample_probs / (jnp.max(pst_sample_probs) * 2)

    # Compute posterior MTTF predictions
    pst_mttf = calc_blacks(
        current_density,
        temps_for_plot,
        prm_samples_posterior['ea'].reshape(-1, 1),
        prm_samples_posterior['A'].reshape(-1, 1),
        BOLTZ_EV,
        prm_samples_posterior['n'].reshape(-1, 1)
    )  # Shape: (num_curves, 100)

    # Plot color-coded MTTF predictions
    sb.set_context('notebook')
    sb.set_theme(style='ticks')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot prior curves
    for i in range(num_curves):
        ax.plot(temp_range_celsius, pri_mttf[i, :], color="#ef00cf", alpha=0.15, linewidth=1.0)

    # Plot posterior curves
    for i in range(num_curves):
        ax.plot(temp_range_celsius, pst_mttf[i, :], color="#6099e5", alpha=0.1, linewidth=1.0)

    # Add legend patches
    prior_patch = pltlines.Line2D([0], [0], color="#ef00cf", linewidth=2, label='Prior predictive MTTF')
    posterior_patch = pltlines.Line2D([0], [0], color="#6099e5", linewidth=2, label='Posterior predictive MTTF')
    ax.legend(handles=[prior_patch, posterior_patch], loc='upper right', fontsize='14')

    ax.set_xlabel('Temperature (°C)', fontsize='18')
    ax.set_ylabel('Mean Time To Failure (hours)', fontsize='18')
    ax.set_yscale('log')
    #ax.set_title('Prior vs Posterior MTTF Predictions\n(Black\'s Equation at constant current density)', 
     #            fontsize='medium', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename_mttf = 'electromigration_mttf_prior_posterior.png'
    plt.savefig(filename_mttf, dpi=150, bbox_inches='tight')
    print(f"\nMTTF prediction plot saved as '{filename_mttf}'")
    plt.show()


if __name__ == '__main__':
    hybrid_bonding_electromigration()



