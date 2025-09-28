#!/usr/bin/env python
# coding: utf-8

# In[5]:


from pathlib import Path
import itertools as it
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# In[6]:


# Combined converter (no global activate/deactivate)
_CONV = ro.default_converter + pandas2ri.converter + numpy2ri.converter


# In[7]:


# Source Init.R that holds the R functions
script = Path("Init.R").resolve()
ro.r['source'](script.as_posix())

# in .py Datei
# script = Path(__file__).with_name("Init.R").resolve()
# ro.r['source'](file=script.as_posix(), chdir=True, encoding="UTF-8")


# In[4]:


# R-Funktionen in Python-Handles überführen
stepwise_plain_r   = ro.globalenv["stepwise_plain"]
stepwise_reduced_r = ro.globalenv["stepwise_reduced"]
stepwise_ff_r      = ro.globalenv["stepwise_ff"]
L0opt_CD_r         = ro.globalenv["L0opt_CD"]
L0opt_CDPSI_r      = ro.globalenv["L0opt_CDPSI"]
Select_GSDAR_r     = ro.globalenv["Select_GSDAR"]


# In[5]:


# Structure of the results 
@dataclass
class ModelSelResult:
    mBIC:   float
    mBIC2:  float
    model1: np.ndarray   # Indizes (1-basiert, wie in R) des mBIC-Optimums
    model2: np.ndarray   # Indizes des mBIC2-Optimums


# In[6]:


def _to_r_numeric_vector(arr):
    """Return an R 'numeric' vector without dimensions."""
    return FloatVector(np.asarray(arr, dtype=float).ravel())


# In[7]:


# Dekorator, der den lokalen Converter ein- und ausschaltet
def _with_conversion(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        with _CONV.context():
            return func(*args, **kwargs)
    return _wrapper


# In[8]:


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


# In[9]:


def lassonet(
        y: np.ndarray,
        X: np.ndarray,
        model: str = "linear", #Default 
        K_max: int | None = None,
) -> ModelSelResult:
    """
    Modellselektion via LassoNet‑Pfad und Bewertung mit mBIC / mBIC2.
    - 'linear'  (default): Gaussian response
    - 'logistic': Binäre Antwort

    Parameters
    ----------
    y, X : numpy‑Arrays mit Zeilen = Beobachtungen, Spalten = Features
    model: "linear" (Gaussian) oder "logistic"

    Returns
    -------
    ModelSelResult
    """
    model = model.lower().strip()
    is_logistic = (model == "logistic")
    if model not in ("linear", "logistic"):
        raise ValueError(f"lassonet_selection: unbekannter model-Typ '{model}'")

    n, p = X.shape                                # Gesamtdatensatz
    if K_max is None:
        K_MAX = int(min(round(p / 2), round(n / 2), 150))
    else:
        K_MAX = int(K_max)

    # --- Split für Log-Likelihood-Schätzung -----------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if is_logistic else None
    )

    n_val = X_val.shape[0]

    # --- LassoNet-Pfad --------------------------------------
    net_cls = LassoNetClassifier if is_logistic else LassoNetRegressor
    net = net_cls(
        hidden_dims=(48,),        # schlanker als (100,)
        path_multiplier=2.3,      # grober, ≈ 6–8 Punkte
        lambda_start="auto",
        n_iters=120,              # max Iter pro Λ
        patience=3,               # Early‑Stopping
        batch_size=256,           # Mini‑Batch‑GD
        val_size=0.15,    # kleineres Val‑Set → kürzere Epochen
        random_state=42
    )

    # Liste aller Checkpoints entlang des Pfads
    #checkpoints = net.path(X_tr, y_tr, return_state_dicts=True)
    # HÖCHSTENS 'max_steps' Pfadpunkte sammeln
    max_steps = 8
    best_mBIC  = np.inf
    best_sup_mBIC  = np.array([], dtype=int)
    best_mBIC2 = np.inf
    best_sup_mBIC2 = np.array([], dtype=int)
    checkpoints = []
    #for idx, state in enumerate(net.path(X_tr, y_tr, return_state_dicts=True)):
    for state in it.islice(net.path(X_tr, y_tr, return_state_dicts=True),
                           max_steps):
        #if idx >= max_steps:
        #    break

        net.load(state)
        support_mask = net.feature_importances_ != 0
        df = int(support_mask.sum())

        checkpoints.append(state)  # <— State sichern

        # ---- Support‑Schranke --------------------------------
        if df > K_MAX:
            break    # Support wächst monoton; spätere wären ≥ K_MAX

        # --- Log-Likelihood auf Val-Set --------------------------------
        if is_logistic:
            proba = net.predict_proba(X_val)[:, 1]
            ll = -log_loss(y_val, proba, normalize=False)
        else:
            pred = net.predict(X_val)
            rss = mean_squared_error(y_val, pred) * len(y_val)
            sigma2 = rss / n_val
            ll = -0.5 * n_val * (np.log(2 * np.pi * sigma2) + 1)

        # --- mBIC & mBIC2 ---------------------------------------------
        penalty = ln(n_val) * df + 2 * ln(p) * df
        mBIC  = -2 * ll + penalty
        mBIC2 = -2 * ll + penalty * ln(ln(n_val))

        # -- mBIC --
        if mBIC < best_mBIC:
            best_mBIC = mBIC
            best_sup_mBIC = np.flatnonzero(support_mask)
        # -- mBIC2 --
        if mBIC2 < best_mBIC2:
            best_mBIC2     = mBIC2
            best_sup_mBIC2 = np.flatnonzero(support_mask)

    # --- Rückgabe ------------------------------------------
    return ModelSelResult(
        mBIC=float(best_mBIC),
        mBIC2=float(best_mBIC2),
        model1=best_sup_mBIC + 1,   # 1-basiert wie R
        model2=best_sup_mBIC2 + 1
    )


# In[10]:


# ---------- Hilfsfunktionen --------------------------------
def _l21_norm(w: torch.Tensor) -> torch.Tensor:
    """Zeilen‑weise L2‑Norm, dann Summe (||·||₂,₁)."""
    return torch.norm(w, dim=1).sum()

# ---------- Hauptfunktion ----------------------------------
def deep2stage(
    y: np.ndarray,
    X: np.ndarray,
    model: str = "linear",          # "linear" | "logistic"
    h: int = 32,                    # Hidden‑Dim der Autoencoder‑Bottleneck‑Schicht
    stage1_epochs: int = 250,
    stage2_epochs: int = 150,
    batch_size: int = 256,
    λ: float = 1.0,                 # Rekonstruktionsgewicht (Stage 1)
    α: float = 5e-3,                # L2,1‑Penalty (Stage 2)
    β: float = 1e-4,                # Frobenius‑Decay (Stage 2)
    lr: float = 1e-3,
    K_max: int | None = None,       # Obergrenze Support‑Größe für mBIC‑Suche
    device: str | torch.device = "cpu",
) -> ModelSelResult:
    """
    Zweiseitiges DNN‑Feature‑Screening gem. Li (2023):
        1. (Supervised) Autoencoder → niedrigdimensionale Repräsentation x_encode
        2. Ein‑Hidden‑Layer‑Netz mit L2,1‑& Frobenius‑Penalty → Feature‑Scores s
    Gibt das mBIC/mBIC2‑Optimum wie die übrigen Wrapper zurück.
    """
    # Test Train Split????
    # X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42,
    #                                        stratify=y if is_logistic else None)
    # Stage‑1/-2 Training auf X_tr, y_tr
    # mBIC‑Berechnung im Loop auf X_val, y_val   # genau wie bei Ihrem LassoNet‑Wrapper

    # -------------- Vorbereitungen --------------------------
    model = model.lower().strip()
    is_logistic = (model == "logistic")
    if model not in ("linear", "logistic"):
        raise ValueError("deep2stage: model muss 'linear' oder 'logistic' sein.")
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float32 if not is_logistic else torch.long,
                          device=device)
    n, p = X.shape
    if K_max is None:
        K_MAX = int(min(round(p / 2), round(n / 2), 150))
    else:
        K_MAX = int(K_max)

    # -------------- Stage 1 – (Supervised) Autoencoder -----
    class SAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(p, 256), nn.ReLU(),
                nn.Linear(256, h)
            )
            self.dec = nn.Sequential(
                nn.Linear(h, 256), nn.ReLU(),
                nn.Linear(256, p)
            )
            # Regressor / Classifier auf Bottleneck
            self.head = nn.Linear(h, 1 if not is_logistic else 2)

        def forward(self, x):
            z = self.enc(x)
            x_hat = self.dec(z)
            y_hat = self.head(z)
            return z, x_hat, y_hat

    sae = SAE().to(device)
    opt1 = optim.RMSprop(sae.parameters(), lr=lr)
    ds = DataLoader(TensorDataset(X_t, y_t), batch_size, shuffle=True)

    for _ in range(stage1_epochs):
        for xb, yb in ds:
            opt1.zero_grad()
            z, x_hat, y_hat = sae(xb)
            recon = nn.functional.mse_loss(x_hat, xb)
            if is_logistic:
                sup = nn.functional.cross_entropy(y_hat, yb)
            else:
                sup = nn.functional.mse_loss(y_hat.squeeze(), yb)
            loss = sup + λ * recon       # Gl. (2)/(3) im Paper :contentReference[oaicite:6]{index=6}
            loss.backward()
            opt1.step()

    with torch.no_grad():
        x_encode = sae.enc(X_t)
        # Normalisieren wie im Paper
        x_encode = (x_encode - x_encode.min(0).values) / (x_encode.max(0).values
                                                          - x_encode.min(0).values + 1e-8)

    # -------------- Stage 2 – Regularisiertes 1‑Hidden‑Net --
    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(p, h)
            self.out = nn.Linear(h, h, bias=True)

        def forward(self, x):
            return self.out(torch.relu(self.fc1(x)))

    stu = Student().to(device)
    opt2 = optim.RMSprop(stu.parameters(), lr=lr)

    for _ in range(stage2_epochs):
        opt2.zero_grad()
        x_hat = stu(X_t)
        mse = nn.functional.mse_loss(x_hat, x_encode)
        l21 = _l21_norm(stu.fc1.weight)          # ||W₁||₂,₁
        frob = (stu.fc1.weight.norm()**2 + stu.out.weight.norm()**2)
        loss = mse + α * l21 + (β / 2) * frob     # Gl. (4) im Paper :contentReference[oaicite:7]{index=7}
        loss.backward()
        opt2.step()

    # -------------- Feature‑Scores & Support ---------------
    with torch.no_grad():
        W1 = stu.fc1.weight.detach()             # h × p
        scores = torch.sum(W1 * W1, dim=0).cpu().numpy()  # s = diag(W₁W₁ᵀ)
    ranked = scores.argsort()[::-1]              # absteigend

    # -------------- mBIC/mBIC2 entlang Top‑k --------------
    best_BIC = best_BIC2 = np.inf
    best_sup_BIC = best_sup_BIC2 = np.array([], dtype=int)

    for k in range(1, min(K_MAX, p) + 1):
        supp = ranked[:k]
        df = k
        # einfache OLS/LogReg‑Fit auf Teilmenge zum LL‑Schätzer
        Xk = X[:, supp]
        if is_logistic:
            from sklearn.linear_model import LogisticRegression
            mdl = LogisticRegression(max_iter=1000).fit(Xk, y)
            ll = mdl.predict_log_proba(Xk).max(axis=1).sum()  # vereinfachter LL
        else:
            from sklearn.linear_model import LinearRegression
            mdl = LinearRegression().fit(Xk, y)
            rss = ((mdl.predict(Xk) - y) ** 2).sum()
            sigma2 = rss / n
            ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)

        penalty = np.log(n) * df + 2 * np.log(p) * df
        mBIC = -2 * ll + penalty
        mBIC2 = -2 * ll + penalty * np.log(np.log(n))

        if mBIC < best_BIC:
            best_BIC, best_sup_BIC = mBIC, supp.copy()
        if mBIC2 < best_BIC2:
            best_BIC2, best_sup_BIC2 = mBIC2, supp.copy()

    return ModelSelResult(
        mBIC=float(best_BIC),
        mBIC2=float(best_BIC2),
        model1=best_sup_BIC + 1,       # 1‑basiert wie die R‑Routinen
        model2=best_sup_BIC2 + 1
    )


# In[22]:


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
    "deep2stage"
]


# In[23]:


get_ipython().system('jupyter nbconvert --to=python "Model_Selection.ipynb" --output=model_selection.py')


# In[ ]:




