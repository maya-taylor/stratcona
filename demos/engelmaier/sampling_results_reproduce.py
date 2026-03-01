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

def calc_engelmaier_sac_snsg(e_f, c_0, c_1, c_2, t_0, T_sj, t_D, delta_D):
    m =  c_0 + c_1*T_sj + c_2*jnp.log(1 + t_0 / t_D) 
    N_f_50 = 0.5*(2*e_f/delta_D)**(1/m)
    return N_f_50

# This function is for recreating the data found in Source #1 since it contained graphs
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
        "c2": -1.72e-02,
        "t_0": 360
    }

    snag_nom = {
        "e_f": 0.275,
        "c_0": 0.430,
        "c1": 6.30e-04,
        "c2": -1.82e-02,
        "t_0": 400
    }

    # variance parameters
    snpb_sigma = {
        "e_f": 0.01,
        "c_0": 0.005,
        "c1":  1e-05,
        "c2":  5e-04,
        "t_0": 5.0
    }

    snag_sigma = {
        "e_f": 0.001,
        "c_0": 0.005,
        "c1":  1e-05,
        "c2":  5e-04,
        "t_0": 5.0
    }

    delta_D = jnp.logspace(-3, 0, 1000)
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

    keys = rand.split(key, n_samples)

    for i in range(n_samples):

        p_snpb = sample_params(keys[i], snpb_nom, snpb_sigma)
        p_snag = sample_params(keys[i], snag_nom, snag_sigma)

        Nf_snpb = calc_engelmaier_sac_snsg(
            p_snpb["e_f"], p_snpb["c_0"], p_snpb["c1"], p_snpb["c2"],
            p_snpb["t_0"], T_sj, t_D, delta_D
        )

        Nf_snag = calc_engelmaier_sac_snsg(
            p_snag["e_f"], p_snag["c_0"], p_snag["c1"], p_snag["c2"],
            p_snag["t_0"], T_sj, t_D, delta_D
        )

        Nf_snpb_all.append(Nf_snpb)
        Nf_snag_all.append(Nf_snag)

    Nf_snpb_all = jnp.stack(Nf_snpb_all)
    Nf_snag_all = jnp.stack(Nf_snag_all)

    # Summary stats
    def summarize(samples):
        mean = jnp.mean(samples, axis=0)
        lower = jnp.percentile(samples, 5, axis=0)
        upper = jnp.percentile(samples, 95, axis=0)
        return mean, lower, upper

    mean_snpb, low_snpb, high_snpb = summarize(Nf_snpb_all)
    mean_snag, low_snag, high_snag = summarize(Nf_snag_all)

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


    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Mean Cycles to Failure (Nf, 50%)")
    plt.ylabel("Inelastic Strain Range (ΔD)")
    plt.title("Engelmaier Model Monte Carlo Bands")

    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()

run_engelmaier_sac_snsg_mc()