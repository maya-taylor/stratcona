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


# the purpose of this script is to test that I am setting
# my priors correctly and the mmtf in reasonable

BOLTZ_EV = 8.617e-5 #boltzmann constant in eV/K

def calc_blacks(current_density, temp, ea, A, k, n):
    mttf = A/pow(current_density, n)*jnp.exp(ea/(k*temp))
    return mttf

A  = 0.5 # to give results of paper roughly
ea = 0.99
n = 1.36
temp = 350 + 273.15
k = BOLTZ_EV
area = 3.6**2 / 1e8 # area in cm^2
current_density = (20E-3/16)/area # 20mA, 16 vias
mttf = calc_blacks(current_density, temp, ea, A, k, n)
print(f"Calculated MTTF: {float(mttf):.6e}")