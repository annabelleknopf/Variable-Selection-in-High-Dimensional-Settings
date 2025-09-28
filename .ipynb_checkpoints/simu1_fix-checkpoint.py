#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, glob, pickle, random, numpy as np
from numpy.random import default_rng
from scipy.linalg import toeplitz
from sklearn.preprocessing import scale
from time import time
import torch

from model_selection import lassonet, lassonet_plus  # ggf. zusätzlich: lassonet_plus

# --------- Pfade / Muster ----------
input_dir  = "Results1_neu"
output_dir = "Results1_update"
file_pattern = "Sim1.k_*.rho_*.pkl"
os.makedirs(output_dir, exist_ok=True)

# --------- Daten-Parameter (bei Bedarf anpassen) ----------
n = 500
p = 1000
p_vec = np.arange(p)

# --------- Repro-Helper (wie im Original) ----------
def make_rng(k, rho, sim, base=19091303):
    return default_rng(base + 100000*k + int(round(10000*rho)) + sim)

def reseed(seed: int, use_torch: bool = False):
    np.random.seed(seed)
    random.seed(seed)
    if use_torch and (torch is not None):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def make_call_seed(k: int, rho: float, sim: int, method_idx: int) -> int:
    rho10 = int(round(10 * rho))
    base = 1_000_000 * k + 10_000 * rho10 + sim
    return int((base * 1_004_659 + method_idx * 97) % 2_147_483_647)

def ar1_cor(p, rho):
    return toeplitz(rho ** np.arange(p))

# Welche Methoden-Spalten sollen ersetzt werden?
TARGETS = {
    "lassonet_plus": (lassonet_plus, True),
}

def process_file(in_path: str):
    with open(in_path, "rb") as f:
        data = pickle.load(f)

    method_names = data["method_names"]
    sim_nr, nr_procedures = data["mBIC_results"].shape
    k   = data["k"]
    rho = data["rho"]

    # zu ersetzende Spalten finden
    target_cols = {}
    for name in TARGETS.keys():
        if name in method_names:
            target_cols[name] = method_names.index(name)

    # Falls keine Zielmethode in der Datei ist: unverändert kopieren
    out_path = os.path.join(output_dir, os.path.basename(in_path))
    if not target_cols:
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(f"[copy] {os.path.basename(in_path)} → {out_path} (keine Zielmethode gefunden)")
        return

    print(f"Update: {os.path.basename(in_path)} | k={k}, rho={rho} | sims={sim_nr}")
    Sigma = ar1_cor(p, rho)

    for sim in range(sim_nr):
        if sim % 10 == 0:
            print(f"  Simulation {sim+1}/{sim_nr}")

        # Daten wie im ursprünglichen Lauf rekonstruieren
        rng = make_rng(k, rho, sim)
        correct_model = rng.choice(p_vec, k, replace=False)
        beta = np.zeros(p)
        beta[correct_model] = rng.normal(0, 1, size=k)

        x = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)
        y = x @ beta + rng.normal(0, 1, size=n)

        X = np.ascontiguousarray(scale(x) / np.sqrt(n), dtype=np.float64)
        y_std = np.ascontiguousarray(scale(y), dtype=np.float64)

        # Zielmethoden neu rechnen
        for name, col_idx in target_cols.items():
            method, uses_torch = TARGETS[name]
            try:
                seed_call = make_call_seed(k, rho, sim, col_idx)
                reseed(seed_call, use_torch=uses_torch)

                t0 = time()
                result = method(y_std, X)
                t1 = time()

                model1 = result.model1 - 1
                model2 = result.model2 - 1

                data["mBIC_results"][sim, col_idx]  = result.mBIC
                data["mBIC2_results"][sim, col_idx] = result.mBIC2
                data["mBIC_FP"][sim, col_idx]       = np.sum(~np.isin(model1, correct_model))
                data["mBIC2_FP"][sim, col_idx]      = np.sum(~np.isin(model2, correct_model))
                data["mBIC_TP"][sim, col_idx]       = np.sum(np.isin(model1, correct_model))
                data["mBIC2_TP"][sim, col_idx]      = np.sum(np.isin(model2, correct_model))
                data["runtime"][sim, col_idx]       = t1 - t0

            except Exception as e:
                print(f"    Fehler '{name}' in Sim {sim+1}: {e} (alte Werte bleiben erhalten)")

    # in neuen Ordner schreiben
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"  → gespeichert in: {out_path}")

# --- Lauf über alle Dateien im Eingabeordner ---
files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
if not files:
    print("Keine passenden Eingabedateien gefunden.")
else:
    for fp in files:
        process_file(fp)

