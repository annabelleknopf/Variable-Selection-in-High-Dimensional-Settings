#!/usr/bin/env python
# coding: utf-8

# In[1]:


###########################################################
#
#   Second set of simulations (GWAS like scenario)
#
###########################################################

# Logistic regression model logit(pi) = X * beta + epsilon

# P(Y = 1) = pi

# Sample size n = 1000, 

# Total number of potential regressors p in c(1000, 2000, 5000, 10000)

# Using independent X (2 allels in Hardy Weinberg equilibrium) 


# One simulation under total null

# For other simulations true model of size k varies between 10 and 20

# Index of variables which enter the model sampled randomly


# Two different scenarios:

# - Coefficients beta of model +- gamma(3,3)

# - Coefficients beta of model N(0,1)


# In[2]:


from __future__ import annotations

import os
import pickle
from time import time

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
    deep2stage_plus,
)


# In[3]:


CheckCode = False  # switch to False for the full run

if CheckCode:
    sim_nr = 2 
    results_folder = "CheckResults2"
else:
    sim_nr = 100
    results_folder = "Results2"

os.makedirs(results_folder, exist_ok=True)

_SCEN2CODE = {"null": 0, "gamma": 1, "normal": 2}

def make_rng(p: int, k: int, scenario: str, sim: int, base: int = 19091302) -> int:
    sc = _SCEN2CODE[scenario]
    s = (base ^ (p * 1_000_003) ^ (k * 9_700_043) ^ (sc * 97_003) ^ (sim * 1_927_211)) & 0x7FFFFFFF
    return s

def reseed(seed: int, use_torch: bool = False):
    """Setzt NumPy/Random (und optional Torch) deterministisch."""
    np.random.seed(seed)
    random.seed(seed)
    if use_torch and (torch is not None):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def make_call_seed(p: int, k: int, scenario: str, sim: int, method_idx: int) -> int:
    """
    Stabiler Seed pro Methoden-Call – abgeleitet aus (p,k,scenario,sim,method_idx).
    Grundlage ist dieselbe Kodierung wie in make_rng(), plus method_idx,
    so dass sich Methoden nicht gegenseitig den RNG-Stream 'weglesen'.
    """
    sc = _SCEN2CODE[scenario]
    base = (p * 1_000_003) ^ (k * 9_700_043) ^ (sc * 97_003) ^ (sim * 1_927_211)
    return int((base * 1_004_659 + method_idx * 97) % 2_147_483_647)


# In[1]:


# Basic dimensions
n = 1_000
p_values = [50_000] #[1_000, 2_000, 5_000, 10_000, 50_000]
k_values = [10,20]    #[10, 20]

# Methods
methods = [
    stepwise_plain,
    L0opt_CD,
    L0opt_CDPSI,
    Select_GSDAR,
    lassonet,
    lassonet_plus,
    deep2stage,
    deep2stage_plus,
]
method_names = [
    "stepwise_plain",
    "L0opt_CD",
    "L0opt_CDPSI",
    "GSDAR", 
    "lassonet",
    "lassonet_plus",
    "deep2stage",
    "deep2stage_plus",
]

TORCH_METHODS = {"lassonet", "lassonet_plus", "deep2stage", "deep2stage_plus"}

nr_procedures = len(methods)
scaler = StandardScaler(with_mean=True, with_std=True)


# In[2]:


def simulate_genotypes(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    """Return an (n × p) genotype matrix with entries in {0, 1, 2}.

    For each marker j we draw its minor‑allele frequency η_j ∼ U(0.1, 0.5).
    Two independent Bernoulli(η_j) alleles per individual are summed, yielding
    X_ij ∼ Binom(2, η_j). This mirrors a simplified GWAS scenario in Hardy–Weinberg equilibrium.
    """

    # η_j for all markers (shape: p)
    maf = rng.uniform(0.1, 0.5, size=p)

    # First and second allele (shape: n × p)
    X0a = rng.binomial(1, maf, size=(n, p))
    X0b = rng.binomial(1, maf, size=(n, p))

    # Additive coding 0/1/2
    return X0a + X0b


# In[4]:


# Helper: unified call that adds the distribution/family argument per method

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


# In[5]:


MAX_ETA = 5.0            # erlaubt |logit| ≤ 5  →  p ∈ (0.007, 0.993)

def simulate_response(rng: np.random.Generator, X_beta: np.ndarray) -> np.ndarray:
    """Generate binary response y ~ Bernoulli(sigmoid(η)) with η bounded."""
    # 1. zentrieren
    η = X_beta - X_beta.mean()

    # 2. nur skalieren, wenn der Vektor nicht konstant 0 ist
    max_abs = np.max(np.abs(η))
    if max_abs > 0:
        η *= MAX_ETA / max_abs        # |η| ≤ MAX_ETA

    # 3. Wahrscheinlichkeiten berechnen
    p = sigmoid(η)
    p = np.clip(p, 1e-12, 1 - 1e-12)  # reine Vorsichtsmaßnahme

    # 4. Binäre Antwort generieren – beide Klassen erzwingen
    y = rng.binomial(1, p)
    if y.min() == y.max():            # passiert nur extrem selten, v.a. bei k = 0
        y = rng.binomial(1, 0.5, size=len(p))

    return y.astype(int)


# In[6]:


# Initialize result matrices
shape = (sim_nr, nr_procedures)
mBIC_results = np.zeros(shape)
mBIC2_results = np.zeros(shape)
mBIC_FP = np.zeros(shape)
mBIC2_FP = np.zeros(shape)
mBIC_TP = np.zeros(shape)
mBIC2_TP = np.zeros(shape)
runtime = np.zeros(shape)

# ----------------------------------------------------------------------------
# Main simulation loop
# ----------------------------------------------------------------------------
for p in p_values:
    p_idx = np.arange(p)

    # =============================
    # Scenario 0: total-null (k=0)
    # =============================
    print(f"Scenario 0 – total null | p = {p}")

    # reset matrices to zero for this scenario
    mBIC_results.fill(np.nan)
    mBIC2_results.fill(np.nan)
    mBIC_FP.fill(np.nan)
    mBIC2_FP.fill(np.nan)
    mBIC_TP.fill(np.nan)
    mBIC2_TP.fill(np.nan)
    runtime.fill(np.nan)

    for sim in range(sim_nr):
        if sim % 10 == 0 or sim == sim_nr - 1:
            print(f"  sim {sim + 1}/{sim_nr}")

        seed = make_rng(p, 0, "null", sim)
        rng = default_rng(seed)

        beta  = np.zeros(p)
        X_raw = simulate_genotypes(rng, n, p)
        y     = simulate_response(rng, X_raw @ beta)
        X     = scaler.fit_transform(X_raw) / np.sqrt(n)

        # Im Null-Szenario gibt es keine wahren Prädiktoren
        correct_model = np.array([], dtype=int)

        for i, method in enumerate(methods):
            try:
                seed_call = make_call_seed(p, 0, "null", sim, i)
                reseed(seed_call, use_torch=(method_names[i] in TORCH_METHODS))

                t0 = time()
                res = run_method(method, method_names[i], y, X)
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


                #print(f" [{method_names[i]}] time={elapsed:.3f}s, mBIC={res.mBIC:.3f}, FP:{mBIC_FP[sim, i]}, TP:{mBIC_TP[sim, i]}")
            except Exception as e:
                print(f"    ⚠️ {method_names[i]} failed: {e}")

    out_path = os.path.join(results_folder, f"Sim2T0.k_0_{p}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "mBIC.results": mBIC_results.copy(),
                "mBIC2.results": mBIC2_results.copy(),
                "mBIC.FP": mBIC_FP.copy(),
                "mBIC2.FP": mBIC2_FP.copy(),
                "mBIC.TP": mBIC_TP.copy(),
                "mBIC2.TP": mBIC2_TP.copy(),
                "runtime": runtime.copy(),
                "method_names": method_names,
                "k": 0,
                "scenario": "null",
                "p": p,
            },
            f,
        )
    print(f"  saved {out_path}")

    # =====================================
    # Scenarios with k ∈ {10,20} true terms
    # =====================================
    for k in k_values:
        for scenario in ("gamma", "normal"):
            label = "a" if scenario == "gamma" else "b"
            print(f"Scenario {label}: k = {k}, p = {p}")

            # reset matrices to zero for this combination
            mBIC_results.fill(0)
            mBIC2_results.fill(0)
            mBIC_FP.fill(0)
            mBIC2_FP.fill(0)
            mBIC_TP.fill(0)
            mBIC2_TP.fill(0)
            runtime.fill(0)

            eff = 10.0
            for sim in range(sim_nr):
                if sim % 10 == 0 or sim == sim_nr - 1:
                    print(f"  sim {sim + 1}/{sim_nr}")

                seed = make_rng(p, k, scenario, sim)
                rng = default_rng(seed)    

                corr_mod = rng.choice(p_idx, k, replace=False)
                if scenario == "gamma":
                    beta_k = rng.choice([-1, 1], size=k) * rng.gamma(3, 1/3, size=k)
                else:
                    beta_k = rng.standard_normal(k)

                beta = np.zeros(p)
                beta[corr_mod] = eff * beta_k

                X_raw = simulate_genotypes(rng, n, p)
                y     = simulate_response(rng, X_raw @ beta)
                X     = scaler.fit_transform(X_raw) / np.sqrt(n)

                # Für den Vergleich mit den wahren Indizes
                correct_model = corr_mod

                for i, method in enumerate(methods):
                    try:
                        seed_call = make_call_seed(p, k, scenario, sim, i)
                        reseed(seed_call, use_torch=(method_names[i] in TORCH_METHODS))

                        t0 = time()
                        res = run_method(method, method_names[i], y, X)
                        elapsed = time() - t0
                        runtime[sim, i] = elapsed

                        # Indizes von R (1-basiert) nach Python (0-basiert)
                        model1 = res.model1 - 1
                        model2 = res.model2 - 1

                        mBIC_results[sim, i]  = res.mBIC
                        mBIC2_results[sim, i] = res.mBIC2
                        mBIC_FP[sim, i]       = np.sum(~np.isin(model1, correct_model))
                        mBIC2_FP[sim, i]      = np.sum(~np.isin(model2, correct_model))
                        mBIC_TP[sim, i]       = np.sum(np.isin(model1, correct_model))
                        mBIC2_TP[sim, i]      = np.sum(np.isin(model2, correct_model))

                        #print(f" [{method_names[i]}] time={elapsed:.3f}, mBIC={res.mBIC:.3f}, FP:{mBIC_FP[sim, i]}, TP:{mBIC_TP[sim, i]} ")
                    except Exception as e:
                        print(f"    ⚠️ {method_names[i]} failed: {e}")

            out_path = os.path.join(results_folder, f"Sim2{label}.k_{k}_{p}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(
                    {
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
                        "p": p,
                    },
                    f,
                )
            print(f"  saved {out_path}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




