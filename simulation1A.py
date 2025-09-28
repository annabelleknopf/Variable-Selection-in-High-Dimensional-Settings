#!/usr/bin/env python
# coding: utf-8

# In[1]:


###########################################################
#
#   First set of simulations (QTL like scenario)
#
#   Additional simulation requested by reviewer
#
###########################################################

# Linear model with Gaussian error  y = X * beta + epsilon

# Sample size n = 500, Total number of potential regressors p = 50000

# Using AR1 for correlation structure of X   (rho = 0.5)
# but only for symmetric band matrix of width 10

# True model of size k = 20

# Index of variables which enter the model sampled randomly

# Coefficients beta of model N(0,1)

# Error term epsilon N(0,1)


# In[2]:


import os
import pickle
from time import time

import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import scale
from scipy.linalg import toeplitz
from scipy.sparse import diags
import random
import torch

import warnings
warnings.filterwarnings(
    "ignore",
    message=r'Environment variable ".*" redefined by R and overriding existing variable\.',
    category=UserWarning,
    module=r"rpy2\.rinterface.*",
)

from model_selection import (
    stepwise_plain,
    stepwise_reduced,
    stepwise_ff,
    L0opt_CD,
    L0opt_CDPSI,
    lassonet,
    lassonet_plus,
    deep2stage,
    deep2stage_plus
)


# In[3]:


CheckCode = True  # set to False for the full simulation

if CheckCode:
    sim_nr = 3        # quick test – “just checking whether the code works”
    p = 5_000
    results_folder = "CheckResults1"
else:
    sim_nr = 100      # original number of simulations
    p = 10_000
    results_folder = "Results1"

os.makedirs(results_folder, exist_ok=True)

def make_rng(k: int, rho: float, sim: int, base: int = 19091303):
    # rho robust in eine ganze Zahl mappen (z.B. 0.5 -> 5000, 0.8 -> 8000)
    return default_rng(base + 100000*k + int(round(10000*rho)) + sim)

def reseed(seed: int, use_torch: bool = False):
    """Setzt NumPy/Random (und optional Torch) deterministisch."""
    np.random.seed(seed)
    random.seed(seed)
    if use_torch and (torch is not None):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def make_call_seed(k: int, rho: float, sim: int, method_idx: int) -> int:
    """Stabiler Seed pro Methoden-Call – unabhängig von hash()."""
    rho10 = int(round(10 * rho))             # 0.0->0, 0.5->5, 0.8->8
    base = 1_000_000 * k + 10_000 * rho10 + sim
    # einfache arithm. Mischung, hält im 31-Bit-Bereich
    return int((base * 1_004_659 + method_idx * 97) % 2_147_483_647)


# In[4]:


# Global parameters

n = 500
k = 20
rho = 0.5
p_vec = np.arange(p)

methods = [
    stepwise_plain,
    stepwise_reduced,
    stepwise_ff,
    L0opt_CD,
    L0opt_CDPSI,
    lassonet,
    lassonet_plus,
    deep2stage,
    deep2stage_plus
]
method_names = [
    "stepwise_plain",
    "stepwise_reduced",
    "stepwise_ff",
    "L0opt_CD",
    "L0opt_CDPSI",
    "lassonet",
    "lassonet_plus",
    "deep2stage",
    "deep2stage_plus"
]
TORCH_METHODS = {"lassonet", "lassonet_plus", "deep2stage", "deep2stage_plus"}
nr_procedures = len(methods)


# In[5]:


# Define correlation structure
def ar1_cor(p: int, rho: float) -> np.ndarray:
    """Dense AR(1) covariance matrix (for small p)."""
    return toeplitz(rho ** np.arange(p))

def band_ar1_cor(p: int, rho: float, width: int = 10):
    """Sparse AR(1) *band* covariance (only first *width* diagonals kept)."""
    diags_data = [rho ** k * np.ones(p - k) for k in range(width)]
    offsets = list(range(width))
    # Build upper band and mirror to lower band in a single call
    return diags(diags_data + diags_data[1:], offsets + [-o for o in offsets[1:]]).toarray()

# Choose covariance representation depending on p (as in R code)
if p < 1_000:
    Sigma = ar1_cor(p, rho)
else:
    Sigma = band_ar1_cor(p, rho)  # width = 10 as specified


# In[6]:


# Initialize result matrices
shape = (sim_nr, nr_procedures)
mBIC_results  = np.full(shape, np.nan)
mBIC2_results = np.full(shape, np.nan)   
mBIC_FP       = np.full(shape, np.nan)
mBIC2_FP      = np.full(shape, np.nan)
mBIC_TP       = np.full(shape, np.nan)
mBIC2_TP      = np.full(shape, np.nan)
runtime       = np.full(shape, np.nan)


# In[7]:


#Simulation

print(
    f"Start: n={n}, p={p}, k={k}, ρ={rho}, sims={sim_nr}, methods={nr_procedures}"
)

for sim in range(sim_nr):
    t_sim = time() #nur zum zeitmessen 
    rng = make_rng(k, rho, sim)
    if sim % 10 == 0 or sim == sim_nr - 1: 
        print(f"  Simulation {sim + 1}/{sim_nr}")

    if sim % 10 == 0 or sim == sim_nr - 1:
        print(f"  Simulation {sim + 1}/{sim_nr}")

    # Generate true model
    correct_model = rng.choice(p_vec, k, replace=False)
    beta = np.zeros(p)
    beta[correct_model] = rng.standard_normal(k)

    # Generate data
    X_raw = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)
    y_raw = X_raw @ beta + rng.standard_normal(n)

    X = np.ascontiguousarray(scale(X_raw) / np.sqrt(n), dtype=np.float64)
    y = np.ascontiguousarray(scale(y_raw), dtype=np.float64)

    # Iterate over methods
    for idx, method in enumerate(methods):
        try:

            seed_call = make_call_seed(k, rho, sim, idx)
            reseed(seed_call, use_torch=(method_names[idx] in TORCH_METHODS))

            t0 = time()
            result = method(y, X)
            elapsed = time() - t0

            model1 = result.model1 - 1  # convert to 0‑based
            model2 = result.model2 - 1

            mBIC_results[sim, idx]  = result.mBIC
            mBIC2_results[sim, idx] = result.mBIC2
            mBIC_FP[sim, idx]       = np.sum(~np.isin(model1, correct_model))
            mBIC2_FP[sim, idx]      = np.sum(~np.isin(model2, correct_model))
            mBIC_TP[sim, idx]       = np.sum(np.isin(model1, correct_model))
            mBIC2_TP[sim, idx]      = np.sum(np.isin(model2, correct_model))
            runtime[sim, idx]       = elapsed


        except Exception as err:
            print(f"    ⚠️  {method_names[idx]} failed in sim {sim + 1}: {err}")

    print(f"✓ Simulation {sim + 1}/{sim_nr} abgeschlossen ({time() - t_sim:.2f}s)", flush=True)
# --------------------------------------------------
# Persist results
# --------------------------------------------------
file_name = os.path.join(results_folder, f"Sim1a.k_{k}.rho_{rho:.1f}.pkl")
with open(file_name, "wb") as f:
    pickle.dump(
        {
            "mBIC_results": mBIC_results,
            "mBIC2_results": mBIC2_results,
            "mBIC_FP": mBIC_FP,
            "mBIC2_FP": mBIC2_FP,
            "mBIC_TP": mBIC_TP,
            "mBIC2_TP": mBIC2_TP,
            "runtime": runtime,
            "method_names": method_names,
            "k": k,
            "rho": rho,
        },
        f,
    )

print(f"Results saved to {file_name}")


# In[ ]:





# In[ ]:




