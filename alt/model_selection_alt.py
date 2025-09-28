#!/usr/bin/env python
# coding: utf-8

# In[83]:


from pathlib import Path
from dataclasses import dataclass
from functools import wraps
import numpy as np
import pandas as pd          
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
import os
from rpy2.robjects.vectors import FloatVector
from lassonet import LassoNetRegressor, LassoNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error
from numpy import log as ln


# In[84]:


# Combined converter (no global activate/deactivate)
_CONV = ro.default_converter + pandas2ri.converter + numpy2ri.converter


# In[85]:


# Source Init.R that holds the R functions
script = Path("Init.R").resolve()
ro.r['source'](script.as_posix())

# in .py Datei
# script = Path(__file__).with_name("Init.R").resolve()
# ro.r['source'](file=script.as_posix(), chdir=True, encoding="UTF-8")


# In[86]:


# R-Funktionen in Python-Handles überführen
stepwise_plain_r   = ro.globalenv["stepwise_plain"]
stepwise_reduced_r = ro.globalenv["stepwise_reduced"]
stepwise_ff_r      = ro.globalenv["stepwise_ff"]
L0opt_CD_r         = ro.globalenv["L0opt_CD"]
L0opt_CDPSI_r      = ro.globalenv["L0opt_CDPSI"]
Select_GSDAR_r     = ro.globalenv["Select_GSDAR"]


# In[87]:


# Structure of the results 
@dataclass
class ModelSelResult:
    mBIC:   float
    mBIC2:  float
    model1: np.ndarray   # Indizes (1-basiert, wie in R) des mBIC-Optimums
    model2: np.ndarray   # Indizes des mBIC2-Optimums


# In[88]:


def _to_r_numeric_vector(arr):
    """Return an R 'numeric' vector without dimensions."""
    return FloatVector(np.asarray(arr, dtype=float).ravel())


# In[89]:


# Dekorator, der den lokalen Converter ein- und ausschaltet
def _with_conversion(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        with _CONV.context():
            return func(*args, **kwargs)
    return _wrapper


# In[90]:


# Python-freundliche Wrapper
@_with_conversion
def stepwise_plain(y: np.ndarray,
                   X: np.ndarray,
                   model: str = "linear") -> ModelSelResult:
    y_r = _to_r_numeric_vector(y)
    res = stepwise_plain_r(y_r, X, model)
    return ModelSelResult(
        mBIC=float(res[0]),
        mBIC2=float(res[1]),
        model1=np.asarray(res[2], dtype=int),
        model2=np.asarray(res[3], dtype=int),
    )

@_with_conversion
def stepwise_reduced(y: np.ndarray,
                     X: np.ndarray,
                     model: str = "linear") -> ModelSelResult:
    y_r = _to_r_numeric_vector(y)
    res = stepwise_reduced_r(y_r, X, model)
    return ModelSelResult(
        mBIC=float(res[0]),
        mBIC2=float(res[1]),
        model1=np.asarray(res[2], dtype=int),
        model2=np.asarray(res[3], dtype=int),
    )

@_with_conversion
def stepwise_ff(y: np.ndarray,
                X: np.ndarray,
                model: str = "linear") -> ModelSelResult:
    y_r = _to_r_numeric_vector(y)
    res = stepwise_ff_r(y_r, X, model)
    return ModelSelResult(
        mBIC=float(res[0]),
        mBIC2=float(res[1]),
        model1=np.asarray(res[2], dtype=int),
        model2=np.asarray(res[3], dtype=int),
    )

#@_with_conversion
#def L0opt_CD(
#        y: np.ndarray,
#        X: np.ndarray,
#        model: str = "SquaredError",
#        maxSuppSize: int | None = None,
#) -> ModelSelResult:
#    y_r = _to_r_numeric_vector(y)
    # ---------- an R weiterreichen ----------
#    res = L0opt_CD_r(y_r, X, model, maxSuppSize=maxSuppSize)
#    return ModelSelResult(
#        mBIC=float(res[0]),
#        mBIC2=float(res[1]),
#        model1=np.asarray(res[2], dtype=int),
#        model2=np.asarray(res[3], dtype=int),
#    )


@_with_conversion
def L0opt_CD(
        y: np.ndarray,
        X: np.ndarray,
        model: str = "SquaredError",
        maxSuppSize: int | None = None,
) -> ModelSelResult:
    y_r = _to_r_numeric_vector(y)

    if maxSuppSize is None:
        # Ohne Argument aufrufen – R nimmt dann maxSuppSize = NA
        res = L0opt_CD_r(y_r, X, model)
    else:
        # Wert explizit übergeben
        res = L0opt_CD_r(y_r, X, model, maxSuppSize=maxSuppSize)

    return ModelSelResult(
        mBIC=float(res[0]),
        mBIC2=float(res[1]),
        model1=np.asarray(res[2], dtype=int),
        model2=np.asarray(res[3], dtype=int),
    )

@_with_conversion
def L0opt_CDPSI(
        y: np.ndarray,
        X: np.ndarray,
        model: str = "SquaredError",
        maxSuppSize: int | None = None,
) -> ModelSelResult:
    y_r = _to_r_numeric_vector(y)

    if maxSuppSize is None:
        res = L0opt_CDPSI_r(y_r, X, model)
    else:
        res = L0opt_CDPSI_r(y_r, X, model, maxSuppSize=maxSuppSize)

    return ModelSelResult(
        mBIC=float(res[0]),
        mBIC2=float(res[1]),
        model1=np.asarray(res[2], dtype=int),
        model2=np.asarray(res[3], dtype=int),
    )

@_with_conversion
def Select_GSDAR(y: np.ndarray,
                 X: np.ndarray) -> ModelSelResult:
    y_r = _to_r_numeric_vector(y)
    res = Select_GSDAR_r(y_r, X)
    return ModelSelResult(
        mBIC=float(res[0]),
        mBIC2=float(res[1]),
        model1=np.asarray(res[2], dtype=int),
        model2=np.asarray(res[3], dtype=int),
    )


# In[95]:


def lassonet(
        y: np.ndarray,
        X: np.ndarray,
        model: str = "linear", #Default 
        k_max: int | None = None
) -> ModelSelResult:
    """
    Modellselektion mit LassoNet.
    - 'linear'  (default): Gaussian response
    - 'logistic': Binäre Antwort

    Gibt mBIC, mBIC2 und 1-basierte Index-Listen der gewählten Features zurück.
    """
    model = model.lower().strip()
    is_logistic = (model == "logistic")
    if model not in ("linear", "logistic"):
        raise ValueError(f"lassonet_selection: unbekannter model-Typ '{model}'")

    # --- Split für Log-Likelihood-Schätzung -----------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if is_logistic else None
    )
    n, p = X_tr.shape

    # --- LassoNet-Pfad --------------------------------------
    net_cls = LassoNetClassifier if is_logistic else LassoNetRegressor
#    net = net_cls(
#        hidden_dims=(100,),
#        path_multiplier=1.3, #1.02
#        lambda_start="auto",  #1.0
#        patience=10
#    )

    net = net_cls(
        hidden_dims=(32,),        # schlanker als (100,)
        path_multiplier=2.3,      # grober, ≈ 6–8 Punkte
        lambda_start="auto",
        #lambda_min_ratio=0.05,    # stoppt Pfad früher
        n_iters=80,              # max Iter pro Λ
        patience=2,               # Early‑Stopping
        batch_size=512,           # Mini‑Batch‑GD
        val_size=0.1            # kleineres Val‑Set → kürzere Epochen
    )


    # Liste aller Checkpoints entlang des Pfads
    #checkpoints = net.path(X_tr, y_tr, return_state_dicts=True)
    # HÖCHSTENS 'max_steps' Pfadpunkte sammeln
    max_steps = 7
    checkpoints = []
    for idx, state in enumerate(
        net.path(X_tr, y_tr, return_state_dicts=True)
    ):
        if idx >= max_steps:
            break
        checkpoints.append(state)

    best_mBIC, best_mBIC2 = np.inf, np.inf
    best_support = np.array([], dtype=int)

    for checkpoint in checkpoints:
        net.load(checkpoint)                 # Netz auf diesen Pfad-Punkt setzen
        # absolute Gewichte der linearen Schicht
        weights = net.feature_importances_          # shape (p,)
        support = np.flatnonzero(weights)           # Indizes ≠ 0
        df = support.size
        if df == 0:
            continue
        if (k_max is not None) and (df > k_max):
            continue                       # zu viele Variablen

        # --- Log-Likelihood auf Val-Set --------------------------------
        if is_logistic:
            proba = net.predict_proba(X_val)[:, 1]
            ll = -log_loss(y_val, proba, normalize=False)
        else:
            pred = net.predict(X_val)
            rss = mean_squared_error(y_val, pred) * len(y_val)
            sigma2 = rss / len(y_val)
            ll = -0.5 * len(y_val) * (np.log(2 * np.pi * sigma2) + 1)

        # --- mBIC & mBIC2 ---------------------------------------------
        penalty = ln(n) * df + 2 * ln(p) * df
        mBIC  = -2 * ll + penalty
        mBIC2 = -2 * ll + penalty * ln(ln(n))

        if mBIC < best_mBIC:
            best_mBIC, best_mBIC2 = mBIC, mBIC2
            best_support = support

    # --- Rückgabe ------------------------------------------
    return ModelSelResult(
        mBIC=best_mBIC,
        mBIC2=best_mBIC2,
        model1=best_support + 1,   # 1-basiert wie R
        model2=best_support + 1
    )


# In[96]:


# Öffentliche API
__all__ = [
    "ModelSelResult",
    "stepwise_plain",
    "stepwise_reduced",
    "stepwise_ff",
    "L0opt_CD",
    "L0opt_CDPSI",
    "Select_GSDAR",
    "lassonet",
]


# In[97]:


get_ipython().system('jupyter nbconvert --to=python "Model_Selection.ipynb" --output=model_selection.py')


# In[ ]:





# In[ ]:




