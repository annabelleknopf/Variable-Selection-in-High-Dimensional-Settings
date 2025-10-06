#!/usr/bin/env python
# coding: utf-8

# In[1]:


###########################################################
#
#   First set of simulations (QTL like scenario)
#
###########################################################

# Linear model with Gaussian error  y = X * beta + epsilon

# Sample size n = 500, Total number of potential regressors p = 1000

# Using AR1 for correlation structure of X   (rho varies between 0, 0.5 and 0.8)

# True model of size k (varies between 0, 5, 10, 20 and 40)

# Index of variables which enter the model sampled randomly

# Coefficients beta of model N(0,1)

# Error term epsilon N(0,1)


# In[2]:


import os
import numpy as np
from numpy.random import default_rng
from scipy.stats import norm
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from scipy.linalg import toeplitz
from time import time
import pickle
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
    ModelSelResult,
    stepwise_plain,
    stepwise_reduced,
    stepwise_ff,
    L0opt_CD,
    L0opt_CDPSI,
    lassonet,
    lassonet_plus,
    deep2stage,
    deep2stage_plus,
)


# In[3]:


# Konfiguration
CheckCode = True
sim_nr = 2 if CheckCode else 500 

# Ergebnisordner abhängig von CheckCode
results_folder = "CheckResults1" if CheckCode else "Results1"
os.makedirs(results_folder, exist_ok=True)  # Ordner wird bei Bedarf angelegt

def make_rng(k, rho, sim, base=19091303):
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
    rho10 = int(round(10 * rho))            
    base = 1_000_000 * k + 10_000 * rho10 + sim
    return int((base * 1_004_659 + method_idx * 97) % 2_147_483_647)


# In[4]:


# Define AR(1) correlation structure
def ar1_cor(p, rho):
    return toeplitz(rho ** np.arange(p))

n = 500
p = 1000
p_vec = np.arange(p)

k_vec = [0, 5, 10, 20, 40] 
rho_vec = [0, 0.5, 0.8]   

# Methodenliste
methods = [
    stepwise_plain,
    stepwise_reduced,
    stepwise_ff,
    L0opt_CD,
    L0opt_CDPSI,
    lassonet,
    lassonet_plus,
    deep2stage,
    deep2stage_plus,
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
    "deep2stage_plus",
]
TORCH_METHODS = {"lassonet", "lassonet_plus", "deep2stage", "deep2stage_plus"} 

nr_procedures = len(methods)
scaler_X = StandardScaler()             
scaler_y = StandardScaler()    
rng = default_rng()


# In[5]:


# Simulation
for k in k_vec:
    for rho in rho_vec:
        # Initialize result matrices
        shape = (sim_nr, nr_procedures)
        mBIC_results  = np.full(shape, np.nan)
        mBIC2_results = np.full(shape, np.nan)
        mBIC_FP       = np.full(shape, np.nan)
        mBIC2_FP      = np.full(shape, np.nan)
        mBIC_TP       = np.full(shape, np.nan)
        mBIC2_TP      = np.full(shape, np.nan)
        runtime       = np.full(shape, np.nan)

        file_name = os.path.join(results_folder, f"Sim1.k_{k}.rho_{rho:.1f}.pkl")
        print(f"Starte Simulation für k={k}, rho={rho} → Datei: {file_name}")

        Sigma = ar1_cor(p, rho)

        for sim in range(sim_nr):

            rng = make_rng(k, rho, sim)

            if sim % 10 == 0:
                print(f"  Simulation {sim + 1} von {sim_nr}")

            # Wahres Modell generieren
            correct_model = rng.choice(p_vec, k, replace=False)
            beta = np.zeros(p)
            beta[correct_model] = rng.normal(0, 1, size=k)

            # X und y generieren
            x = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)
            y = x @ beta + rng.normal(0, 1, size=n)

            # Skalierung
            X = np.ascontiguousarray(scale(x) / np.sqrt(n), dtype=np.float64)
            y = np.ascontiguousarray(scale(y), dtype=np.float64)


            # Methoden durchlaufen
            for idx, method in enumerate(methods):
                try:
                    seed_call = make_call_seed(k, rho, sim, idx)
                    reseed(seed_call, use_torch=(method_names[idx] in TORCH_METHODS))

                    start = time()
                    result = method(y, X)
                    end = time()

                    # Indizes von R nach Python konvertieren (0-basiert)
                    model1 = result.model1 - 1
                    model2 = result.model2 - 1

                    mBIC_results[sim, idx]  = result.mBIC
                    mBIC2_results[sim, idx] = result.mBIC2
                    mBIC_FP[sim, idx]       = np.sum(~np.isin(model1, correct_model))
                    mBIC2_FP[sim, idx]      = np.sum(~np.isin(model2, correct_model))
                    mBIC_TP[sim, idx]       = np.sum(np.isin(model1, correct_model))
                    mBIC2_TP[sim, idx]      = np.sum(np.isin(model2, correct_model))
                    runtime[sim, idx]       = end - start

                except Exception as e:
                    print(f"Fehler bei Methode {method_names[idx]} in Simulation {sim + 1}: {e}")

        # Ergebnisse speichern
        with open(file_name, 'wb') as f:
            pickle.dump({
                'mBIC_results': mBIC_results,
                'mBIC2_results': mBIC2_results,
                'mBIC_FP': mBIC_FP,
                'mBIC2_FP': mBIC2_FP,
                'mBIC_TP': mBIC_TP,
                'mBIC2_TP': mBIC2_TP,
                'runtime': runtime,
                'method_names': method_names,
                'k': k,
                'rho': rho
            }, f)

        print(f"Gespeichert: {file_name}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




