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

# This script is for reproducing the sampling results for the Engelmaier model that in the paper
# listed in source 1 below. The goal was to check that I had reasonable priors to help produce good results
# once I used them in the actual model.


def calc_engelmaier_sac_snsg(e_f, c_0, c_1, c_2, t_0, T_sj, t_D, delta_D):
    m =  c_0 + c_1*T_sj + c_2*jnp.log(1 + t_0 / t_D) 
    N_f_50 = 0.5*jnp.pow(2*e_f/delta_D)(1/m)
    return N_f_50

# This function is for recreating the data found in Source #1 since it contained graphs
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

def run_engelmaier_sac_snsg_mc(
    n_samples=500,
    seed=0
):

    key = rand.PRNGKey(seed)

    # Nominal values
    snpb_nom = {
        "e_f": 0.325,
        "c_0": 0.442,
        "c1": 6.00e-04,
        "c2": -1.74e-02,
        "t_0": 360
    }

    # variance parameters
    snpb_sigma = {
        "e_f": 0.01,
        "c_0": 0.005,
        "c1":  1e-05,
        "c2":  5e-04,
        "t_0": 5.0
    }

    snag_nom = {
        "e_f": 0.275,
        "c_0": 0.430,
        "c1": 6.30e-04,
        "c2": -1.82e-02,
        "t_0": 400
    }

    snag_sigma = {
        "e_f": 0.001,
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

    delta_D = jnp.logspace(-4, 0, 1000)
    T_sj = 50
    t_D = 600

    def sample_params(key, nom_dict, sigma_dict):
        keys = rand.split(key, len(nom_dict))
        sampled = {}
        for i, k in enumerate(nom_dict):
            mu = nom_dict[k]
            sigma = sigma_dict[k]
            sampled[k] = mu + sigma * rand.normal(keys[i])
        return sampled

    Nf_snpb_all = []
    Nf_snag_all = []
    Nf_sac105_all = []

    keys = rand.split(key, n_samples)

    for i in range(n_samples):

        p_snpb = sample_params(keys[i], snpb_nom, snpb_sigma)
        p_snag = sample_params(keys[i], snag_nom, snag_sigma)
        p_sac105 = sample_params(keys[i], sac105_nom, sac105_sigma)

        Nf_snpb = calc_engelmaier_sac_snsg(
            p_snpb["e_f"], p_snpb["c_0"], p_snpb["c1"], p_snpb["c2"],
            p_snpb["t_0"], T_sj, t_D, delta_D
        )

        Nf_snag = calc_engelmaier_sac_snsg(
            p_snag["e_f"], p_snag["c_0"], p_snag["c1"], p_snag["c2"],
            p_snag["t_0"], T_sj, t_D, delta_D
        )

        Nf_sac105 = calc_engelmaier_sac_snsg(
            p_sac105["e_f"], p_sac105["c_0"], p_sac105["c1"], p_sac105["c2"],
            p_sac105["t_0"], T_sj, t_D, delta_D
        )

        Nf_snpb_all.append(Nf_snpb)
        Nf_snag_all.append(Nf_snag)
        Nf_sac105_all.append(Nf_sac105)

    Nf_snpb_all = jnp.stack(Nf_snpb_all)
    Nf_snag_all = jnp.stack(Nf_snag_all)
    Nf_sac105_all = jnp.stack(Nf_sac105_all)

    # Summary stats
    def summarize(samples):
        mean = jnp.mean(samples, axis=0)
        lower = jnp.percentile(samples, 5, axis=0)
        upper = jnp.percentile(samples, 95, axis=0)
        return mean, lower, upper

    mean_snpb, low_snpb, high_snpb = summarize(Nf_snpb_all)
    mean_snag, low_snag, high_snag = summarize(Nf_snag_all)
    mean_sac105, low_sac105, high_sac105 = summarize(Nf_sac105_all)

    # Plot
    plt.figure(figsize=(8,6))

    # ---- SnPb ----
    plt.plot(
        mean_snpb,
        100*delta_D,
        color="blue",
        linewidth=2,
        label="SnPb Mean"
    )

    plt.fill_betweenx(
        100*delta_D,
        low_snpb,
        high_snpb,
        color="blue",
        alpha=0.2
    )

    plt.plot(low_snpb, 100*delta_D, color="blue", linestyle="--", linewidth=1)
    plt.plot(high_snpb, 100*delta_D, color="blue", linestyle="--", linewidth=1)


    # ---- SnAg ----
    plt.plot(
        mean_snag,
        100*delta_D,
        color="red",
        linewidth=2,
        label="SnAg Mean"
    )

    plt.fill_betweenx(
        100*delta_D,
        low_snag,
        high_snag,
        color="red",
        alpha=0.2
    )

    plt.plot(low_snag, 100*delta_D, color="red", linestyle=":", linewidth=1)
    plt.plot(high_snag, 100*delta_D, color="red", linestyle=":", linewidth=1)

    # ---- SAC105 ----
    plt.plot(
        mean_sac105,
        100*delta_D,
        color="green",
        linewidth=2,
        label="SAC105 Mean"
    )

    plt.fill_betweenx(
        100*delta_D,
        low_sac105,
        high_sac105,
        color="green",
        alpha=0.2
    )

    plt.plot(low_sac105, 100*delta_D, color="green", linestyle="-.", linewidth=1)
    plt.plot(high_sac105, 100*delta_D, color="green", linestyle="-.", linewidth=1)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(10**1, 10**6)

    plt.xlabel("Mean Cycles to Failure (Nf, 50%)")
    plt.ylabel("Inelastic Strain Range (ΔD)")
    plt.title("Engelmaier Model")

    # Add labels at Nf = 10^1, 10^2, 10^3, 10^4, 10^5, 10^6
    for power in range(1, 7):
        nf_target = 10.0 ** power
        # Find closest index in mean_snpb
        idx_snpb = jnp.argmin(jnp.abs(mean_snpb - nf_target))
        # Plot SnPb label as point coordinates
        delta_d_value = 100*delta_D[idx_snpb]
        plt.text(mean_snpb[idx_snpb], delta_d_value, f'  ($10^{{{power}}}$, {delta_d_value:.3f}%)', 
                fontsize=9, color="blue", va='center', ha='left')
        
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()

run_engelmaier_sac_snsg_mc()