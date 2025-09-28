#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fügt nachträglich die Methode `stepwise_ff` in bestehende Simulations-Pickles ein,
ohne die übrigen Methoden neu zu berechnen.

Beispiel:
    python add_stepwise_ff_to_pickles.py --folder Results2 --out-folder Results2_stepwiseFF
"""

import os
import sys
import argparse
import pickle
import time
import warnings
import random

import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler

# Deine Implementationen
from model_selection import stepwise_ff

warnings.filterwarnings("ignore")

# ---------- Konstanten wie im Originalskript ----------
N = 1000                   # Sample Size (aus deinem Skript)
MAX_ETA = 5.0              # Begrenzung für den Logit
EFF = 10.0                 # Effektgröße bei k>0
SCALER = StandardScaler(with_mean=True, with_std=True)

_SCEN2CODE = {"null": 0, "gamma": 1, "normal": 2}

def make_rng(p: int, k: int, scenario: str, sim: int, base: int = 19091302) -> int:
    sc = _SCEN2CODE[scenario]
    s = (base ^ (p * 1_000_003) ^ (k * 9_700_043) ^ (sc * 97_003) ^ (sim * 1_927_211)) & 0x7FFFFFFF
    return s

def make_call_seed(p: int, k: int, scenario: str, sim: int, method_idx: int) -> int:
    sc = _SCEN2CODE[scenario]
    base = (p * 1_000_003) ^ (k * 9_700_043) ^ (sc * 97_003) ^ (sim * 1_927_211)
    return int((base * 1_004_659 + method_idx * 97) % 2_147_483_647)

def reseed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

# ---------- Data-Gen wie im Originalskript ----------
def simulate_genotypes(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    maf = rng.uniform(0.1, 0.5, size=p)
    X0a = rng.binomial(1, maf, size=(n, p))
    X0b = rng.binomial(1, maf, size=(n, p))
    return X0a + X0b

def simulate_response(rng: np.random.Generator, X_beta: np.ndarray) -> np.ndarray:
    eta = X_beta - X_beta.mean()
    max_abs = np.max(np.abs(eta))
    if max_abs > 0:
        eta *= MAX_ETA / max_abs
    p = 1.0 / (1.0 + np.exp(-eta))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    y = rng.binomial(1, p)
    if y.min() == y.max():
        y = rng.binomial(1, 0.5, size=len(p))
    return y.astype(int)

def run_stepwise_ff(y, X):
    # Alle stepwise*-Methoden bekommen model="logistic"
    return stepwise_ff(y, X, model="logistic")

def _to_zero_based(arr) -> np.ndarray:
    if arr is None:
        return np.array([], dtype=int)
    a = np.array(arr, dtype=int)
    if a.size == 0:
        return a
    return a - 1

def process_pickle(path: str, out_folder: str | None = None):
    with open(path, "rb") as f:
        d = pickle.load(f)

    # Erwartete Struktur prüfen (minimal)
    required = ["mBIC.results", "mBIC2.results", "mBIC.FP", "mBIC2.FP",
                "mBIC.TP", "mBIC2.TP", "runtime", "method_names", "k", "scenario", "p"]
    for key in required:
        if key not in d:
            print(f"[skip] {os.path.basename(path)}: Schlüssel fehlt: {key}")
            return

    method_names = list(d["method_names"])
    if "stepwise_ff" in method_names:
        print(f"[skip] {os.path.basename(path)} enthält bereits stepwise_ff.")
        return

    p = int(d["p"])
    k = int(d["k"])
    scenario = str(d["scenario"])  # 'null', 'gamma', 'normal'
    runtime_arr = np.array(d["runtime"])
    if runtime_arr.ndim != 2:
        print(f"[skip] {os.path.basename(path)}: runtime hat unerwartete Form {runtime_arr.shape}")
        return

    sims, m = runtime_arr.shape  # sim_nr, nr_procedures
    print(f"[{os.path.basename(path)}] p={p}, k={k}, scenario={scenario}, sims={sims}")

    # Neue Spalten anlegen
    new_mBIC   = np.full((sims, 1), np.nan)
    new_mBIC2  = np.full((sims, 1), np.nan)
    new_FP1    = np.full((sims, 1), np.nan)
    new_FP2    = np.full((sims, 1), np.nan)
    new_TP1    = np.full((sims, 1), np.nan)
    new_TP2    = np.full((sims, 1), np.nan)
    new_time   = np.full((sims, 1), np.nan)

    method_idx = m  # wir hängen hinten an → stabiler Seed pro Methode
    p_idx = np.arange(p)

    for sim in range(sims):
        seed = make_rng(p, k, scenario, sim)
        rng = default_rng(seed)

        # Wahres Modell & Koeffizienten reproduzieren
        if k == 0 or scenario == "null":
            correct_model = np.array([], dtype=int)
            beta = np.zeros(p)
        else:
            corr_mod = rng.choice(p_idx, k, replace=False)
            if scenario == "gamma":
                beta_k = rng.choice([-1, 1], size=k) * rng.gamma(3, 1/3, size=k)
            elif scenario == "normal":
                beta_k = rng.standard_normal(k)
            else:
                print(f"  ⚠️ Unbekanntes Szenario: {scenario}")
                return
            beta = np.zeros(p)
            beta[corr_mod] = EFF * beta_k
            correct_model = corr_mod

        # Daten generieren (identische Reihenfolge wie im Original!)
        X_raw = simulate_genotypes(rng, N, p)
        y = simulate_response(rng, X_raw @ beta)
        X = SCALER.fit_transform(X_raw) / np.sqrt(N)

        # Stabiler Seed für die Methode (falls intern RNG genutzt)
        reseed(make_call_seed(p, k, scenario, sim, method_idx))

        try:
            t0 = time.time()
            res = run_stepwise_ff(y, X)
            elapsed = time.time() - t0
            new_time[sim, 0] = elapsed

            model1 = _to_zero_based(getattr(res, "model1", []))
            model2 = _to_zero_based(getattr(res, "model2", []))

            # Kennzahlen
            new_mBIC[sim, 0]  = float(getattr(res, "mBIC", np.nan))
            new_mBIC2[sim, 0] = float(getattr(res, "mBIC2", np.nan))

            new_FP1[sim, 0] = int(np.sum(~np.isin(model1, correct_model))) if model1.size else 0
            new_FP2[sim, 0] = int(np.sum(~np.isin(model2, correct_model))) if model2.size else 0
            new_TP1[sim, 0] = int(np.sum(np.isin(model1, correct_model))) if model1.size else 0
            new_TP2[sim, 0] = int(np.sum(np.isin(model2, correct_model))) if model2.size else 0

        except Exception as e:
            print(f"  ⚠️ sim {sim+1}/{sims}: stepwise_ff fehlgeschlagen: {e}")

    # ---- speichern: in neuen Ordner oder originalen Pfad ----
    if out_folder:
        os.makedirs(out_folder, exist_ok=True)
        dest = os.path.join(out_folder, os.path.basename(path))
    else:
        # Optionales Backup, nur wenn wir überschreiben
        bak = path + ".bak"
        if not os.path.exists(bak):
            try:
                os.rename(path, bak)
            except Exception:
                pass
        dest = path

    # Spalten anhängen
    d["mBIC.results"]  = np.concatenate([d["mBIC.results"],  new_mBIC], axis=1)
    d["mBIC2.results"] = np.concatenate([d["mBIC2.results"], new_mBIC2], axis=1)
    d["mBIC.FP"]       = np.concatenate([d["mBIC.FP"],       new_FP1], axis=1)
    d["mBIC2.FP"]      = np.concatenate([d["mBIC2.FP"],      new_FP2], axis=1)
    d["mBIC.TP"]       = np.concatenate([d["mBIC.TP"],       new_TP1], axis=1)
    d["mBIC2.TP"]      = np.concatenate([d["mBIC2.TP"],      new_TP2], axis=1)
    d["runtime"]       = np.concatenate([d["runtime"],       new_time], axis=1)
    d["method_names"]  = method_names + ["stepwise_ff"]

    with open(dest, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  ✅ gespeichert: {dest}")

def main():
    ap = argparse.ArgumentParser(description="stepwise_ff nachträglich in Simulations-Pickles einfügen.")
    ap.add_argument("--folder", default="Results2",
                    help="Eingabe-Ordner mit Simulations-Pickles (Default: Results2)")
    ap.add_argument("--out-folder", default=None,
                    help="Ausgabe-Ordner für neue Pickles (optional; überschreibt sonst Originale)")
    args = ap.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        print(f"Ordner nicht gefunden: {folder}", file=sys.stderr)
        sys.exit(1)

    files = [os.path.join(folder, fn) for fn in os.listdir(folder) if fn.endswith(".pkl")]
    if not files:
        print("Keine .pkl-Dateien gefunden.")
        return

    for path in sorted(files):
        process_pickle(path, out_folder=args.out_folder)

if __name__ == "__main__":
    main()
