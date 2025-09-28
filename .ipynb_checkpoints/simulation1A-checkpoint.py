#!/usr/bin/env python
# coding: utf-8

# In[11]:


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


# In[12]:


import os
import pickle
from time import time

import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import scale
from scipy.linalg import toeplitz
from scipy.sparse import diags

from model_selection import (
    stepwise_plain,
    stepwise_reduced,
    stepwise_ff,
    L0opt_CD,
    L0opt_CDPSI,
    lassonet,
    deep2stage,
    lassonetm_fast,
    lassonetm_quality
)


# In[13]:


CheckCode = True  # set to False for the full simulation

if CheckCode:
    sim_nr = 3        # quick test – “just checking whether the code works”
    p = 5_000
    results_folder = "CheckResults1"
else:
    sim_nr = 100      # original number of simulations
    p = 50_000
    results_folder = "Results1"

os.makedirs(results_folder, exist_ok=True)

def make_rng(k: int, rho: float, sim: int, base: int = 19091303):
    # rho robust in eine ganze Zahl mappen (z.B. 0.5 -> 5000, 0.8 -> 8000)
    return default_rng(base + 100000*k + int(round(10000*rho)) + sim)


# In[14]:


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
    deep2stage,
    lassonetm_fast,
    lassonetm_quality
]
method_names = [
    "stepwise_plain",
    "stepwise_reduced",
    "stepwise_ff",
    "L0opt_CD",
    "L0opt_CDPSI",
    "lassonet",
    "deep2stage",
    "lassonetm_fast",
    "lassonetm_quality"
]

nr_procedures = len(methods)


# In[15]:


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


# In[16]:


# Initialize result matrices
shape = (sim_nr, nr_procedures)

mBIC_results = np.zeros(shape)
mBIC2_results = np.zeros(shape)
mBIC_FP       = np.zeros(shape)
mBIC2_FP      = np.zeros(shape)
mBIC_TP       = np.zeros(shape)
mBIC2_TP      = np.zeros(shape)
runtime       = np.zeros(shape)


# In[17]:


#Simulation

print(
    f"Start: n={n}, p={p}, k={k}, ρ={rho}, sims={sim_nr}, methods={nr_procedures}"
)

for sim in range(sim_nr):
    rng = make_rng(k, rho, sim)

    if sim % 10 == 0 or sim == sim_nr - 1:
        print(f"  Simulation {sim + 1}/{sim_nr}")

    # Generate true model
    correct_model = rng.choice(p_vec, k, replace=False)
    beta = np.zeros(p)
    beta[correct_model] = rng.standard_normal(k)

    # Generate data
    X_raw = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)
    y_raw = X_raw @ beta + rng.standard_normal(n)

    X = scale(X_raw) / np.sqrt(n)
    y = scale(y_raw)

    # Iterate over methods
    for idx, method in enumerate(methods):
        try:
            t0 = time()
            result = method(y, X)
            elapsed = time() - t0

            print(f"    {method_names[idx]}: {elapsed:8.3f} s")

            model1 = result.model1 - 1  # convert to 0‑based
            model2 = result.model2 - 1

            mBIC_results[sim, idx]  = result.mBIC
            mBIC2_results[sim, idx] = result.mBIC2
            mBIC_FP[sim, idx]       = np.sum(~np.isin(model1, correct_model))
            mBIC2_FP[sim, idx]      = np.sum(~np.isin(model2, correct_model))
            mBIC_TP[sim, idx]       = np.sum(np.isin(model1, correct_model))
            mBIC2_TP[sim, idx]      = np.sum(np.isin(model2, correct_model))
            runtime[sim, idx]       = elapsed

            print({result.mBIC})


        except Exception as err:
            print(f"    ⚠️  {method_names[idx]} failed in sim {sim + 1}: {err}")

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




