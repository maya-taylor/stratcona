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

##################################################################
# This script is for generating the observed data for thermomech.py
#################################################################

def generate_data():
    """Generate 6 data points for three different test conditions"""
    
    def calc_delta_D(e_f, c_0, c_1, c_2, t_0, T_sj, t_D, N_f_target):
        """Solve for delta_D given target N_f: delta_D = 2*e_f / (2*N_f)^m"""
        m = c_0 + c_1*T_sj + c_2*jnp.log(1 + t_0 / t_D)
        delta_D = 2*e_f / jnp.power(2*N_f_target, m)
        return delta_D
    
    # SnPb reference parameters
    snpb_nom = {
        "e_f": 0.325,
        "c_0": 0.442,
        "c_1": 6.00e-04,
        "c_2": -1.74e-02,
        "t_0": 360
    }
    
    # Target Nf values: 10^1 to 10^6
    target_Nf = jnp.array([1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    
    # Condition 1: T_sj=50, t_D=600
    print("Condition 1: T_sj=50, t_D=600")
    delta_D_cond1 = []
    for nf in target_Nf:
        delta_d = calc_delta_D(
            snpb_nom["e_f"], snpb_nom["c_0"], snpb_nom["c_1"], snpb_nom["c_2"],
            snpb_nom["t_0"], T_sj=50, t_D=600, N_f_target=nf
        )
        delta_D_cond1.append(float(delta_d))
        print(f"  N_f={nf:.0e} -> delta_D={float(delta_d):.6f}")
    
    # Condition 2: T_sj=150, t_D=600
    print("\nCondition 2: T_sj=150, t_D=600")
    delta_D_cond2 = []
    for nf in target_Nf:
        delta_d = calc_delta_D(
            snpb_nom["e_f"], snpb_nom["c_0"], snpb_nom["c_1"], snpb_nom["c_2"],
            snpb_nom["t_0"], T_sj=150, t_D=600, N_f_target=nf
        )
        delta_D_cond2.append(float(delta_d))
        print(f"  N_f={nf:.0e} -> delta_D={float(delta_d):.6f}")
    
    # Condition 3: T_sj=40, t_D=300
    print("\nCondition 3: T_sj=40, t_D=300")
    delta_D_cond3 = []
    for nf in target_Nf:
        delta_d = calc_delta_D(
            snpb_nom["e_f"], snpb_nom["c_0"], snpb_nom["c_1"], snpb_nom["c_2"],
            snpb_nom["t_0"], T_sj=40, t_D=300, N_f_target=nf
        )
        delta_D_cond3.append(float(delta_d))
        print(f"  N_f={nf:.0e} -> delta_D={float(delta_d):.6f}")
    
    print("\nData for thermomech.py:")
    print("Condition 1: T_sj=50, t_D=600")
    print(f"delta_D_Nfdata_cond1 = {[[delta_D_cond1[i], float(target_Nf[i])] for i in range(len(target_Nf))]}")
    print("\nCondition 2: T_sj=150, t_D=600")
    print(f"delta_D_Nfdata_cond2 = {[[delta_D_cond2[i], float(target_Nf[i])] for i in range(len(target_Nf))]}")
    print("\nCondition 3: T_sj=40, t_D=300")
    print(f"delta_D_Nfdata_cond3 = {[[delta_D_cond3[i], float(target_Nf[i])] for i in range(len(target_Nf))]}")


if __name__ == '__main__':
    generate_data()
    