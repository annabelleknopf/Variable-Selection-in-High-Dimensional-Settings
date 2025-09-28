#!/usr/bin/env python
# coding: utf-8

# In[1]:


###########################################################
#
#   Fourth set of simulations (Real GWAS data for X)
#
###########################################################

# Logistic regression model logit(pi) = X * beta + epsilon

# P(Y = 1) = pi

# Sample size n = 1000, 

# Total number of potential regressors p = 7297


# Coefficients beta of model from gamma distribution


# Simulation scenarios where causal markers are distributed 
# differently among clusters of SNPs:


# 1)  k = 20 causal SNPs distributed over singletons

# 2)  k = 20 causal SNPs distributed over Clusters of at least size 5
#            1 SNP per cluster      
#

# 3)  k = 20 causal SNPs distributed over Clusters of at least size 5
#            2 SNPs per cluster      
#

# 4)  k = 20 causal SNPs distributed over Clusters of at least size 5
#            4 SNPs per cluster      
#


# In[2]:


import os
import pickle
from time import time
from typing import List

import numpy as np
from numpy.random import default_rng, Generator
from scipy.special import expit as sigmoid
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import scale
import pyreadr
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
    L0opt_CD,
    L0opt_CDPSI,
    Select_GSDAR,
    lassonet,
    lassonet_plus,
    deep2stage,
    deep2stage_plus
)


# In[3]:


CheckCode = False  # switch to False for the full run

if CheckCode:
    sim_nr = 2
    results_folder = "CheckResults4"
else:
    sim_nr = 100
    results_folder = "Results4"

os.makedirs(results_folder, exist_ok=True)

RNG_SEED = 19091303
k = 20                      # number of true SNPs in every scenario
eff = 15.0                  # effect scaling factor from R code

def reseed(seed: int, use_torch: bool = False):
    """Setzt NumPy/Random (und optional Torch) deterministisch."""
    np.random.seed(seed)
    random.seed(seed)
    if use_torch and (torch is not None):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def make_call_seed(scenario: int, sim: int, method_idx: int, base: int = 19091303) -> int:
    """
    Stabiler Seed für *einen* Methoden-Call – abgeleitet aus (scenario, sim, method_idx).
    So beeinflussen sich Methoden nicht gegenseitig über den RNG-Stream.
    """
    # scenario: 1..4 (dein Mapping)
    val = (base ^ (scenario * 97_003) ^ (sim * 1_927_211) ^ (method_idx * 101))
    return int(val & 0x7FFF_FFFF)


# In[4]:


# Load SNP data
r_data_path = os.path.join("Data", "SNP_data.RData")
if not os.path.exists(r_data_path):
    raise FileNotFoundError(f"{r_data_path} not found – please provide the file.")

r_objects = pyreadr.read_r(r_data_path)
X_SNP = r_objects["X_SNP"].to_numpy()
#Cluster_list = r_objects["Cluster.list"]  # list of 1‑based vectors (R style)
import pandas as _pd  # local import
_df: _pd.DataFrame = r_objects["Cluster.df"]
# first column = cluster ID, second column = SNP index (1‑based)
clust_col, snp_col = _df.columns[:2]
_grouped = _df.groupby(clust_col)[snp_col]
Cluster_list = [grp.values for _, grp in _grouped]
#print(f"→ Converted Cluster.df with {len(Cluster_list)} clusters.")

# Convert clusters (1‑based R indices) to 0‑based NumPy arrays (1‑based R indices) to 0‑based NumPy arrays
Cluster_list_py = [np.asarray(cl, dtype=int) - 1 for cl in Cluster_list]


n, p = X_SNP.shape
scaler = StandardScaler(with_mean=True, with_std=True)
X = np.ascontiguousarray(scaler.fit_transform(X_SNP) / np.sqrt(n), dtype=np.float64)

# SNP categories
Singletons = np.concatenate([cl for cl in Cluster_list_py if len(cl) == 1]).astype(int)
LargerClusters = [cl.astype(int) for cl in Cluster_list_py if len(cl) >= 5]


# In[5]:


# Helper: unified call that adds the correct signature per method
def run_method(method, method_name: str, y, X):
    """Call *method* with the appropriate signature mirroring the R code."""
    if method_name.startswith("stepwise"):
        # stepwise_plain / _reduced / _ff
        return method(y, X, model="logistic")
    if method_name == "L0opt_CD":
        return method(y, X, model="Logistic", maxSuppSize=50)
    if method_name == "L0opt_CDPSI":
        return method(y, X, model="Logistic", maxSuppSize=50)
    if method_name == "GSDAR":
        return method(y, X) 
    if method_name == "lassonet":
        return method(y, X, model="logistic") 
    if method_name == "lassonet_plus":
        return method(y, X, model="logistic") 
    if method_name == "deep2stage":
        return method(y, X, model="logistic") 
    if method_name == "deep2stage_plus":
        return method(y, X, model="logistic") 
    # fallback – should not occur
    return method(y, X)


# In[6]:


MAX_ETA = 5.0        # erlaubt |logit| ≤ 5  →  p ∈ (0.007, 0.993)
def simulate_response(rng: Generator, X_beta: np.ndarray) -> np.ndarray:
    """Generate binary response y ~ Bernoulli(sigmoid(η)) with η bounded."""

    # 1. zentrieren
    eta = X_beta - X_beta.mean()

    # 2. nur skalieren, wenn der Vektor nicht konstant 0 ist
    max_abs = np.max(np.abs(eta))
    if max_abs > 0:
        eta *= MAX_ETA / max_abs      # |η| ≤ MAX_ETA

    # 3. Wahrscheinlichkeiten berechnen
    p_vec = sigmoid(eta)
    p_vec = np.clip(p_vec, 1e-12, 1 - 1e-12)

    # 4. Binäre Antwort generieren
    y = rng.binomial(1, p_vec)
    # Guarantee both classes (rare edge case)
    if y.min() == y.max():
        y = rng.binomial(1, 0.5, size=len(y))
    return y.astype(int)


# In[7]:


# Helper: pick causal SNP indices ----------------------------------------

def pick_causal_snps(scenario: int, rng: Generator) -> np.ndarray:
    if scenario == 1:
        return rng.choice(Singletons, k, replace=False)

    snps_per_cluster = {2: 1, 3: 2, 4: 4}[scenario]
    clusters_needed = k // snps_per_cluster
    cluster_idx = rng.choice(len(LargerClusters), clusters_needed, replace=False)
    causal = []
    for idx in cluster_idx:
        causal.extend(rng.choice(LargerClusters[idx], snps_per_cluster, replace=False))
    return np.asarray(causal, dtype=int)


# In[8]:


methods = [stepwise_plain, Select_GSDAR, L0opt_CD, L0opt_CDPSI, lassonet, lassonet_plus, deep2stage, deep2stage_plus]
method_names = [
    "stepwise_plain",
    "GSDAR",
    "L0opt_CD",
    "L0opt_CDPSI",
    "lassonet",
    "lassonet_plus",
    "deep2stage",
    "deep2stage_plus"
]
TORCH_METHODS = {"lassonet", "lassonet_plus", "deep2stage", "deep2stage_plus"}
nr_procedures = len(methods)


# In[9]:


# Results matrices

shape = (sim_nr, nr_procedures)
mBIC_results = np.zeros(shape)
mBIC2_results = np.zeros(shape)
mBIC_FP = np.zeros(shape)
mBIC2_FP = np.zeros(shape)
mBIC_TP = np.zeros(shape)
mBIC2_TP = np.zeros(shape)
runtime = np.zeros(shape)


# In[10]:


# Main simulation loop 

rng = default_rng(RNG_SEED)

for scenario in (4,):  #(1, ) (1, 2, 3, 4)
    scen_label = f"Scen{scenario}"
    print(f"\nScenario {scenario}: {sim_nr} simulation(s)…")

    # Reset matrices for this scenario
    for arr in (mBIC_results, mBIC2_results, mBIC_FP, mBIC2_FP, mBIC_TP, mBIC2_TP, runtime):
        arr.fill(np.nan)

    for sim in range(sim_nr):
        if sim % 10 == 0 or sim == sim_nr - 1:
            print(f"  sim {sim + 1}/{sim_nr}")

        correct = pick_causal_snps(scenario, rng)
        beta_k = rng.choice([-1, 1], k) * rng.gamma(3, 1/3, k)
        beta = np.zeros(p)
        beta[correct] = eff * beta_k

        y = simulate_response(rng, X @ beta)

        for i, (method, name) in enumerate(zip(methods, method_names)):
            try:
                seed_call = make_call_seed(scenario=scenario, sim=sim, method_idx=i)
                reseed(seed_call, use_torch=(name in TORCH_METHODS))

                t0 = time()
                res = run_method(method, name, y, X)
                runtime[sim, i] = time() - t0

                m1 = np.asarray(res.model1, dtype=int) - 1
                m2 = np.asarray(res.model2, dtype=int) - 1

                mBIC_results[sim, i]  = res.mBIC
                mBIC2_results[sim, i] = res.mBIC2
                mBIC_FP[sim, i]       = np.sum(~np.isin(m1, correct))
                mBIC2_FP[sim, i]      = np.sum(~np.isin(m2, correct))
                mBIC_TP[sim, i]       = np.sum(np.isin(m1, correct))
                mBIC2_TP[sim, i]      = np.sum(np.isin(m2, correct))

            except Exception as e:
                print(f"    ⚠️ {name} failed: {e}")

    out_file = os.path.join(results_folder, f"{scen_label}.k_{k}.pkl")
    with open(out_file, "wb") as f:
        pickle.dump({
            "mBIC.results": mBIC_results.copy(),
            "mBIC2.results": mBIC2_results.copy(),
            "mBIC.FP": mBIC_FP.copy(),
            "mBIC2.FP": mBIC2_FP.copy(),
            "mBIC.TP": mBIC_TP.copy(),
            "mBIC2.TP": mBIC2_TP.copy(),
            "runtime": runtime.copy(),
            "method_names": method_names,
            "k": k,
            "scenario": scenario,
            "n": n,
            "p": p,
        }, f)
    print(f"  → saved {out_file}")

print("\nAll scenarios completed.")


# In[ ]:





# In[ ]:





# In[ ]:




