#!/usr/bin/env python
# coding: utf-8

# In[6]:


#  REAL DATA - eQTL mapping

#
#  Data set is available at ftp://ftp.sanger.ac.uk/pub/genevar/
#
#  For convenience we provide the file Sangerdata.Rdata which contains
#  all the data already in an Rdata file which can be found in the "Data" folder.
#


# In[7]:


import os
import pickle
import numpy as np
from time import time
import torch
import pyreadr

from sklearn.preprocessing import scale

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
    lassonet,
    lassonet_plus,
    deep2stage,
    deep2stage_plus
)


# In[8]:


# --- Konfiguration ---
results_folder = "ResultsRealData"
os.makedirs(results_folder, exist_ok=True)

def reseed(seed: int, use_torch: bool = False):
    """Setzt NumPy/Random (und optional Torch) deterministisch."""
    np.random.seed(seed)
    import random as _random
    _random.seed(seed)
    if use_torch and (torch is not None):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # optional deterministischer machen:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False

def make_call_seed(method_idx: int, gene_row: int, base: int = 19091303) -> int:
    """
    Stabiler Seed pro Methoden-Call – abgeleitet aus (gene_row, Methode).
    So bleibt es reproduzierbar, auch wenn du später ein anderes Gen nimmst.
    """
    val = (base ^ (gene_row * 1009) ^ (method_idx * 101))
    return int(val & 0x7FFF_FFFF)


# In[9]:


# Pfad zur R-Data-Datei (relativ zum Skript-Standort)
rdata_path = os.path.join("Data", "Sangerdata.Rdata")

# --- Daten einlesen (pyreadr) ---
# liefert ein dict: Schlüssel sind die Objekt-Namen im R-Environment
rdata = pyreadr.read_r(rdata_path)
data = rdata["data"]

# R-Indizes sind 1-basiert, in Python 0-basiert:
gene_row = 24266 - 1

# Data from gene CCT8 corresponds to line 24266
# y: Ausprägung für Gen CCT8
y = data.iloc[gene_row, 1:].astype(float).values

# x: alle anderen Zeilen, ohne erste Spalte, transponiert
x = data.drop(index=gene_row).iloc[:, 1:].astype(float).values.T

# Skalierung analog zu R: X nach Spalten skalieren und durch sqrt(n) teilen
n, p = x.shape
X = scale(x) / np.sqrt(n)
y = scale(y)


# In[10]:


# --- Methodenliste (ohne stepwise_reduced & stepwise_ff) ---
methods = [stepwise_plain, L0opt_CDPSI, L0opt_CD, lassonet, lassonet_plus, deep2stage, deep2stage_plus]
method_names = ["stepwise_plain", "L0opt_CDPSI", "L0opt_CD", "lassonet", "lassonet_plus", "deep2stage", "deep2stage_plus"]
TORCH_METHODS = {"lassonet", "lassonet_plus", "deep2stage", "deep2stage_plus"}
# Ergebnis-Container
results = {}
runtimes = {}


# In[8]:


for i, (method, name) in enumerate(zip(methods, method_names)):
    print(f"Starte Methode: {name}")
    seed_call = make_call_seed(method_idx=i, gene_row=gene_row)
    reseed(seed_call, use_torch=(name in TORCH_METHODS))

    start = time()
    result = method(y, X)
    end = time()

    results[name] = result
    runtimes[name] = end - start

# --- Speichern ---
out_path = os.path.join(results_folder, "RealData_eQTL.pkl")
with open(out_path, "wb") as f:
    pickle.dump({
        "results": results,
        "runtimes": runtimes,
        "method_names": method_names,
        "n": n,
        "p": p
    }, f)

print(f"Alle Ergebnisse gespeichert in: {out_path}")


# In[ ]:





# In[ ]:




