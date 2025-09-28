#!/usr/bin/env python
# coding: utf-8

# In[1]:


###########################################################
#
#   Third set of Simulations: GWAS‑style design with AR(1) LD blocks
#
###########################################################
# Logistic regression model:   logit(π) = X · β
# Binary outcome:              Y ~ Bernoulli(π)
#
# Sample size                n = 1 000
# Candidate predictors       p ∈ {1 000, 2 000, 5 000, 10 000, 50 000}
#
# Design matrix  X
#   • Continuous, z‑standardised covariates – think SNP dosages after
#     imputation rather than hard 0/1/2 counts.
#   • Blocks of varying length
#   • Within each block an AR(1) / Toeplitz correlation structure
#       Corr(X_i, X_j) = ρ^{|i‑j|}.
#   • ρ is a deterministic function of block size (0.85 for ≤10 SNP, …,
#     0.45 for >40 SNP) – mirrors the empirical decay of linkage
#     disequilibrium (LD) with physical distance
#
# Simulation scenarios
#   0) *Total null*          : β = 0 (controls Type‑I error)
#   a) Sparse Gamma effects  : k ∈ {10,20} non‑zero β_j  ~ ± 10 Gamma(3,1/3)
#   b) Sparse Normal effects : k ∈ {10,20} non‑zero β_j  ~ 10 N(0,1)
#
# Recorded performance metrics   per method × simulation run
#   • mBIC       /  mBIC2
#   • False positives  (mBIC_FP   / mBIC2_FP)
#   • True  positives  (mBIC_TP   / mBIC2_TP)
#   • Runtime in seconds
#
# Methods under comparison:
# stepwise_plain, L0opt_CD, L0opt_CDPSI, Select_GSDAR, lassonet


# In[2]:


from __future__ import annotations

import os
import pickle
from time import time
from typing import List

import numpy as np
from numpy.random import default_rng, Generator
from scipy.special import expit as sigmoid
from sklearn.preprocessing import StandardScaler

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


# -----------------------------------------------------------------------------
#  Global switches
# -----------------------------------------------------------------------------
CheckCode = False  # switch to False for the full run

if CheckCode:
    sim_nr = 1  
    results_folder = "CheckResults3"
else:
    sim_nr = 100
    results_folder = "Results3"

os.makedirs(results_folder, exist_ok=True)

_SCEN2CODE = {"null": 0, "gamma": 1, "normal": 2}

def make_rng_ld(p: int, k: int, scenario: str, sim: int, base: int = 12345671):
    sc = _SCEN2CODE[scenario]
    seed = (base + 1_000_003*p + 97_003*k + 1_003*sc + sim) % (2**32 - 1)
    return default_rng(int(seed))

def reseed(seed: int, use_torch: bool = False):
    """Setzt NumPy/Random (und optional Torch) deterministisch für den Methoden-Call."""
    np.random.seed(seed)
    random.seed(seed)
    if use_torch and (torch is not None):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def make_call_seed(p: int, k: int, scenario: str, sim: int, method_idx: int) -> int:
    """
    Stabiler Seed für *einen* Methoden-Call – abgeleitet aus (p, k, scenario, sim, Methode).
    Damit beeinflussen sich die Methoden nicht gegenseitig über den RNG-Stream.
    """
    sc = _SCEN2CODE[scenario]
    base = (p * 1_000_003) ^ (k * 9_700_043) ^ (sc * 97_003) ^ (sim * 1_927_211)
    return int((base * 1_004_659 + method_idx * 97) % 2_147_483_647)


# In[4]:


# -----------------------------------------------------------------------------
#  Basic dimensions
# -----------------------------------------------------------------------------


n = 1_000
p_values = [1000]  #[1_000, 2_000, 5_000, 10_000, 50_000]
k_values = [10, 20]

# -----------------------------------------------------------------------------
#  Block definitions per p (adjust freely)
# -----------------------------------------------------------------------------

# ρ‑Heuristik: <=10 → 0.85, 11‑20 → 0.75, 21‑30 → 0.65, 31‑40 → 0.55, >40 → 0.45

_rho = lambda s: 0.85 if s <= 10 else 0.75 if s <= 20 else 0.65 if s <= 30 else 0.55 if s <= 40 else 0.45

BLOCK_STRUCTURES = {
    1_000:  {"sizes": [10]*20 + [15]*10 + [25]*10 + [50]*8},
    2_000:  {"sizes": [10]*20 + [20]*25 + [30]*25 + [40]*10 + [50]*3},
    5_000:  {"sizes": [40]*10 + [60]*30 + [100]*20 + [200]*4},
    10_000: {"sizes": [10]*200 + [15]*100 + [25]*100 + [50]*80},
    50_000: {"sizes": [10_000]*2 + [5_000]*2 + [2_500]*4 + [1_000]*10},
}

for spec in BLOCK_STRUCTURES.values():
    spec["rhos"] = [_rho(s) for s in spec["sizes"]]


# In[5]:


# -----------------------------------------------------------------------------
#  Covariate generator – AR(1) inside each block
# -----------------------------------------------------------------------------


def _ar1_block(rng: Generator, n: int, m: int, rho: float) -> np.ndarray:
    """Generate (n × m) matrix with AR(1) correlation rho^{|i-j|} using N(0,1) noise."""
    eps = rng.standard_normal((n, m))  # Var=1
    X = np.empty((n, m))
    X[:, 0] = eps[:, 0]
    coef = np.sqrt(1 - rho**2)         # sorgt für stationäre Varianz=1
    for j in range(1, m):
        X[:, j] = rho * X[:, j - 1] + coef * eps[:, j]
    return X

def simulate_covariates(rng: Generator, n: int, blocks: List[int], rhos: List[float]) -> np.ndarray:
    parts = [_ar1_block(rng, n, m, rho) for m, rho in zip(blocks, rhos)]
    return np.hstack(parts)


# In[6]:


# -----------------------------------------------------------------------------
#  Helper: logit‑response simulation (unchanged)
# -----------------------------------------------------------------------------

MAX_ETA = 5.0  # ⇒ P(Y=1) ∈ (0.7%, 99.3%)


def simulate_response(rng: Generator, X_beta: np.ndarray) -> np.ndarray:
    """Generate binary response with bounded logits."""
    eta = X_beta - X_beta.mean()
    max_abs = np.max(np.abs(eta))
    if max_abs > 0:
        eta *= MAX_ETA / max_abs

    p = sigmoid(eta)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    y = rng.binomial(1, p)

    # ensure both classes occur (rare but possible for k=0)
    if y.min() == y.max():
        y = rng.binomial(1, 0.5, size=len(p))

    return y.astype(int)


# In[7]:


# -----------------------------------------------------------------------------
#  Unified call wrapper per selection method
# -----------------------------------------------------------------------------

def run_method(method, method_name: str, y, X):
    if method_name.startswith("stepwise"):
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
    return method(y, X)


# In[8]:


# -----------------------------------------------------------------------------
#  Simulation driver
# -----------------------------------------------------------------------------

methods = [
    stepwise_plain,
    L0opt_CD,
    L0opt_CDPSI,
    Select_GSDAR,
    lassonet,
    lassonet_plus,
    deep2stage,
    deep2stage_plus

]
method_names = [
    "stepwise_plain",
    "L0opt_CD",
    "L0opt_CDPSI",
    "GSDAR",
    "lassonet",
    "lassonet_plus",
    "deep2stage",
    "deep2stge_plus"
]

TORCH_METHODS = {"lassonet", "lassonet_plus", "deep2stage", "deep2stge_plus"}
nr_procedures = len(methods)
scaler = StandardScaler(with_mean=True, with_std=True)

RNG_SEED = 19091302
rng: Generator = default_rng(RNG_SEED)

# result matrices 
shape = (sim_nr, nr_procedures)

mBIC_results = np.zeros(shape)
mBIC2_results = np.zeros(shape)
mBIC_FP = np.zeros(shape)
mBIC2_FP = np.zeros(shape)
mBIC_TP = np.zeros(shape)
mBIC2_TP = np.zeros(shape)
runtime = np.zeros(shape)


# In[9]:


# -----------------------------------------------------------------------------
#  Main simulation loop
# -----------------------------------------------------------------------------

for p in p_values:
    sizes = BLOCK_STRUCTURES[p]["sizes"]
    rhos  = BLOCK_STRUCTURES[p]["rhos"]
    idx_all = np.arange(p)

    # ------------------------------- Scenario 0 ------------------------------
    print(f"\nScenario 0 (null) | p={p}")
    for arr in (mBIC_results, mBIC2_results, mBIC_FP, mBIC2_FP, mBIC_TP, mBIC2_TP, runtime):
        arr.fill(np.nan)

    for sim in range(sim_nr):
        if sim % 10 == 0 or sim == sim_nr - 1:
            print(f"  sim {sim + 1}/{sim_nr}")

        rng = make_rng_ld(p, 0, "null", sim)

        X_raw = simulate_covariates(rng, n, sizes, rhos)
        y = simulate_response(rng, np.zeros(n))
        X = np.ascontiguousarray(scaler.fit_transform(X_raw) / np.sqrt(n), dtype=np.float64)

        # Im Null-Szenario sind keine wahren Prädiktoren aktiv
        correct_model = np.array([], dtype=int)

        for i, (mtd, name) in enumerate(zip(methods, method_names)):
            try:
                seed_call = make_call_seed(p, 0, "null", sim, i)
                reseed(seed_call, use_torch=(name in {"lassonet","deep2stage","lassonetm_fast","lassonetm_quality"}))

                t0 = time()
                res = run_method(mtd, name, y, X)
                elapsed = time() - t0
                runtime[sim, i] = elapsed

                # Indizes von R (1-basiert) nach Python (0-basiert)
                model1 = res.model1 - 1
                model2 = res.model2 - 1

                mBIC_results[sim, i]  = res.mBIC
                mBIC2_results[sim, i] = res.mBIC2
                mBIC_FP[sim, i]       = np.sum(~np.isin(model1, correct_model))
                mBIC2_FP[sim, i]      = np.sum(~np.isin(model2, correct_model))
                # TP=0 im Null-Szenario
                mBIC_TP[sim, i]       = 0
                mBIC2_TP[sim, i]      = 0

            except Exception as e:
                print(f"    ⚠️ {name} failed: {e}")

    fname = os.path.join(results_folder, f"Sim3T0.k_0_{p}.pkl")
    with open(fname, "wb") as f_out:
        pickle.dump(dict(mBIC_results=mBIC_results, mBIC2_results=mBIC2_results,
                         mBIC_FP=mBIC_FP, mBIC2_FP=mBIC2_FP,
                         mBIC_TP=mBIC_TP, mBIC2_TP=mBIC2_TP,
                         runtime=runtime, method_names=method_names,
                         k=0, scenario="null", p=p), f_out)
    print(f"  saved {fname}")

    # ------------------------- Signal scenarios a,b --------------------------
    for k in k_values:
        for scenario in ("gamma", "normal"):
            label = "a" if scenario == "gamma" else "b"
            print(f"Scenario {label} | k={k} | p={p}")

            for arr in (mBIC_results, mBIC2_results, mBIC_FP, mBIC2_FP, mBIC_TP, mBIC2_TP, runtime):
                arr.fill(np.nan)

            eff = 10.0
            for sim in range(sim_nr):
                if sim % 10 == 0 or sim == sim_nr - 1:
                    print(f"  sim {sim + 1}/{sim_nr}")

                rng = make_rng_ld(p, k, scenario, sim) 

                true_idx = rng.choice(idx_all, k, replace=False)
                if scenario == "gamma":
                    beta_k = rng.choice([-1, 1], size=k) * rng.gamma(3, 1/3, size=k)
                else:
                    beta_k = rng.standard_normal(k)

                beta = np.zeros(p)
                beta[true_idx] = eff * beta_k

                X_raw = simulate_covariates(rng, n, sizes, rhos)
                y = simulate_response(rng, X_raw @ beta)
                X = np.ascontiguousarray(scaler.fit_transform(X_raw) / np.sqrt(n), dtype=np.float64)

                correct_model = true_idx

                for i, (mtd, name) in enumerate(zip(methods, method_names)):
                    try:
                        seed_call = make_call_seed(p, k, scenario, sim, i)
                        reseed(seed_call, use_torch=(name in TORCH_METHODS))

                        t0 = time()
                        res = run_method(mtd, name, y, X)
                        elapsed = time() - t0
                        runtime[sim, i] = elapsed

                        model1 = res.model1 - 1
                        model2 = res.model2 - 1

                        mBIC_results[sim, i]  = res.mBIC
                        mBIC2_results[sim, i] = res.mBIC2
                        mBIC_FP[sim, i]       = np.sum(~np.isin(model1, correct_model))
                        mBIC2_FP[sim, i]      = np.sum(~np.isin(model2, correct_model))
                        mBIC_TP[sim, i]       = np.sum(np.isin(model1, correct_model))
                        mBIC2_TP[sim, i]      = np.sum(np.isin(model2, correct_model))

                    except Exception as e:
                        print("⚠️", name, "failed:", e)
            out_sig = os.path.join(results_folder, f"Sim3_{scenario}.k{k}_p{p}.pkl")
            with open(out_sig, "wb") as f_out:
                pickle.dump(dict(mBIC_results=mBIC_results, mBIC2_results=mBIC2_results,
                                 mBIC_FP=mBIC_FP, mBIC2_FP=mBIC2_FP,
                                 mBIC_TP=mBIC_TP, mBIC2_TP=mBIC2_TP,
                                 runtime=runtime, method_names=method_names,
                                 k=k, scenario=scenario, p=p), f_out)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




