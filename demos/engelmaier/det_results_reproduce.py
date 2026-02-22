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

# This document is for finding appropriate priors for modelling the Engelmaier Equation
# Plan:
# - For each data source, use their values and see when kind of Nf values I get and if they are reasonable



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
def calc_engelmaier_sac_snsg(e_f, c_0, c_1, c_2, t_0, T_sj, t_D, delta_D):
    m =  c_0 + c_1*T_sj + c_2*jnp.log(1 + t_0 / t_D) 
    N_f_50 = 0.5*(2*e_f/delta_D)**(1/m)
    return N_f_50

# This function is for recreating the data found in Source #1 since it contained graphs
# I will first try to recreate it deterministically
def run_engelmaier_sac_snsg_det():
    # For recreating the data, I want to sweep sheer strain from 0.01 to 100
    # and record the mean cycles to failure

    # Variables for SnPb
    e_f_snpb     = 0.325
    c_0_snpb     = 0.442
    c1_snpb      = 6.00e-04
    c2_snpb      = -1.72e-02
    t_0_snpb     = 360
    
    # Variable for SnAg
    e_f_snag     = 0.275
    c_0_snag     = 0.430
    c1_snag      = 6.30e-04
    c2_snag      = -1.82e-02
    t_0_snag     = 400

    # Global variables (this is not a great name)
    delta_D = jnp.logspace(-3, 0, 1000)  # generating the sample space of strain values
    T_sj    = 50 # this is in Celsius 
    t_D     = 600 # says 10 minutes on the graph

    Nf_snpb = calc_engelmaier_sac_snsg(
        e_f_snpb, c_0_snpb, c1_snpb, c2_snpb,
        t_0_snpb, T_sj, t_D, delta_D
    )

    Nf_snag = calc_engelmaier_sac_snsg(
        e_f_snag, c_0_snag, c1_snag, c2_snag,
        t_0_snag, T_sj, t_D, delta_D
    )

    # Plot strain (ΔD) on y-axis
    plt.figure()
    plt.plot(Nf_snpb, 100*delta_D, label="SnPb")
    plt.plot(Nf_snag, 100*delta_D, label="SnAg")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Mean Cycles to Failure (Nf, 50%)")
    plt.ylabel("Inelastic Strain Range (ΔD)")
    plt.title("Engelmaier Model: Strain vs Cycles to Failure")

    plt.legend()
    plt.grid(True, which="both")
    plt.show()
    return

# run_engelmaier_sac_snsg_det()


# Source #2 https://ieeexplore.ieee.org/abstract/document/10835367
# Fatigue Life Analysis of Hybrid Bonding in Micro-LED Interconnection
# Parameter for this equation:
# - N_f_50: the number to cycles to failure at 50% probability (median fatigue life)
# - e_f: fatigue ductility coefficient (solder material property)
# - delta_e: range of equivalent plastic strain
# - c_0, c_1, c_2 : empirical found material coefficients
# - T_A: average temperature of the thermal cycling (C)
# - f : frequency of cycling (cycles per day)

def calc_engelmaier_hybrid_microled(e_f, delta_e, c_0, c_1, c_2, T_a, f):
    c =  c_0 + c_1*T_a + c_2*jnp.log(1 + f) 
    print("constant c = ", float(c))
    N_f_50 = 0.5*((jnp.sqrt(3) / 2) * ( delta_e / e_f) )**(-1/c) # this paper uses -1/c? 
    
    
    return N_f_50

# This function is for recreating the data found in Source #1 since it contained graphs
# I will first try to recreate it deterministically
def run_engelmaier_hybrid_microled():
    # For recreating the data, I want to sweep sheer strain from 0.01 to 100
    # and record the mean cycles to failure

    # Variables for Hybrid Bonding MicroLED Paper
        # This paper does not disclose the true constants
    e_f     = 0.325
    delta_e = 6.708e-3
    c_0     = 0.3552  #0.422orig tweaking to match the c = 0.397 they report
    c_1     = -6e-4 
    c_2     = 1.74e-2 #twea
    T_a     = 35
    f       = 35

    Nf_hybrid_microled = calc_engelmaier_hybrid_microled(
        e_f, delta_e, c_0, c_1, c_2, T_a, f
    )

    print(Nf_hybrid_microled)
   
    return

run_engelmaier_hybrid_microled()