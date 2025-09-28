#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.linear_model import LinearRegression, LogisticRegression
from numpy import log as ln
import itertools as it
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings(
    "ignore",
    message=r'Environment variable ".*" redefined by R and overriding existing variable\.',
    category=UserWarning,
    module=r"rpy2\.rinterface.*",
)


# In[2]:


# Combined converter (no global activate/deactivate)
_CONV = ro.default_converter + pandas2ri.converter + numpy2ri.converter


# In[3]:


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
mbic_r  = ro.globalenv["mbic_py"]
mbic2_r = ro.globalenv["mbic2_py"]


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


@_with_conversion
def _mbic_bigstep(loglik: float, n: int, k: int, p: int, const: float = 4.0) -> float:
    return float(np.asarray(mbic_r(loglik, n, k, p, const=const))[0])

@_with_conversion
def _mbic2_bigstep(loglik: float, n: int, k: int, p: int, const: float = 4.0) -> float:
    return float(np.asarray(mbic2_r(loglik, n, k, p, const=const))[0])


# In[ ]:


# Stabilität des Trainings
def _prepare_for_lassonet(X, y):

    # 1) near-zero-Var-Spalten entschärfen (verhindert Grad/Weight-Explosionen)
    s = X.std(axis=0, ddof=0)
    near_zero = s < 1e-12
    if near_zero.any():
        X = X.copy()
        X[:, near_zero] = 0.0  # konstant setzen

    # 2) Einheitlicher Dtype für Torch
    X = X.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)

    return X, y


# In[9]:


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


# In[23]:


def lassonet(
    y: np.ndarray,
    X: np.ndarray,
    model: str = "linear",
    *,
    M: int | None = None,
    path_multiplier: float | None = None,
    hidden_dims: tuple[int, ...] | None = None,
    n_iters: int | None = None,
    early_patience: int | None = None,
    no_improve_patience: int | None = None,
    random_state: int | None = None,
    device: str | None = None,
) -> ModelSelResult:
    """
    LassoNet-Pfad auswerten und bestes Modell (mBIC/mBIC2) wählen.
    Verbesserungssignal = (mBIC besser) ODER (k = |Support| sinkt).
    Abbruch nach 'no_improve_patience' aufeinanderfolgenden Supports ohne Verbesserungssignal.
    """
    # ---- Defaults (wie bisher) ----
    NO_IMPROVE_PATIENCE = 4 if no_improve_patience is None else no_improve_patience
    TOL = 1e-9
    SAT_DELTA = 5
    HIDDEN_DIMS = (4,) if hidden_dims is None else hidden_dims
    PATH_MULTIPLIER = 1.08 if path_multiplier is None else path_multiplier
    N_ITERS = 100 if n_iters is None else n_iters
    EARLY_PATIENCE = 2 if early_patience is None else early_patience
    RANDOM_STATE = 42 if random_state is None else random_state
    M = 30 if M is None else M
    DEVICE = "cpu" if device is None else device
    BATCH_SIZE = min(X.shape[0], 512)

    # ---- Vorbereitungen ----
    model = model.lower().strip()
    if model not in ("linear", "logistic"):
        raise ValueError("lassonet: model must be 'linear' or 'logistic'.")
    is_logistic = (model == "logistic")

    n, p = X.shape
    SAT_CUTOFF = min(int(p/2),int(n/2),150)

    def _ll_full(supp: np.ndarray) -> tuple[float, int]:
        """Log-Likelihood für gegebenen Support."""
        k = supp.size
        if k == 0:
            if is_logistic:
                p_hat = float(y.mean())
                p_hat = min(max(p_hat, 1e-12), 1 - 1e-12)
                ll = float((y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat)).sum())
            else:
                sigma2_0 = max(float(np.var(y, ddof=0)), 1e-12)
                ll = -0.5 * n * np.log(sigma2_0)
            return ll, 0

        Xk = X[:, supp]
        if is_logistic:
            mdl = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(Xk, y.astype(int))
            proba = np.clip(mdl.predict_proba(Xk)[:, 1], 1e-12, 1 - 1e-12)
            ll = float((y * np.log(proba) + (1 - y) * np.log(1 - proba)).sum())
        else:
            mdl = LinearRegression().fit(Xk, y)
            rss = float(((mdl.predict(Xk) - y) ** 2).sum())
            sigma2 = max(rss / n, 1e-12)
            ll = -0.5 * n * np.log(sigma2)
        return ll, k

    net_cls = LassoNetClassifier if is_logistic else LassoNetRegressor
    net = net_cls(
        hidden_dims=HIDDEN_DIMS,
        path_multiplier=PATH_MULTIPLIER,
        lambda_start="auto",
        n_iters=N_ITERS,
        patience=EARLY_PATIENCE,
        batch_size=BATCH_SIZE,
        val_size=0.0,
        random_state=RANDOM_STATE,
        M=M,
        device=DEVICE,
        optim=(torch.optim.Adam, torch.optim.Adam),
    )
    # Vorher: X war bereits scale(x)/sqrt(n); wir hatten wieder *sqrt(n) gemacht.
    # Für Stabilität beim Netz: bewusst *nicht* zurückskalieren.
    X_net, y_net = _prepare_for_lassonet(X, y)
    X_path = X_net
    y_fit  = y_net

    #X_path = X * float(np.sqrt(n))
    #y_fit = y.astype(int) if is_logistic else y

    best_mbic = float("inf");   best_sup_mbic = np.array([], dtype=int);   patience_mbic = 0
    best_mbic2 = float("inf");  best_sup_mbic2 = np.array([], dtype=int);  patience_mbic2 = 0
    best_k_for_mbic  = np.inf
    best_k_for_mbic2 = np.inf

    seen_supports: set[tuple[int, ...]] = set()

    for _, ckpt in enumerate(net.path(X_path, y_fit)):
        selected = getattr(ckpt, "selected", getattr(ckpt, "selected_features_", None))
        if selected is None:
            continue

        mask = np.asarray(selected, dtype=bool)
        df = int(mask.sum())
        if df >= SAT_CUTOFF:
            continue

        supp = np.flatnonzero(mask)
        key = tuple(supp.tolist())
        if key in seen_supports:
            continue
        seen_supports.add(key)

        ll, k = _ll_full(supp)
        mbic  = _mbic_bigstep(ll, n, k, p)
        mbic2 = _mbic2_bigstep(ll, n, k, p)

        # mBIC
        improved_val = (mbic < best_mbic - TOL)
        improved_k   = (k < best_k_for_mbic)
        if improved_val or improved_k:
            if improved_val:
                best_mbic, best_sup_mbic = mbic, supp.copy()
            best_k_for_mbic = min(best_k_for_mbic, k)
            patience_mbic = 0
        else:
            patience_mbic += 1

        # mBIC2
        improved_val2 = (mbic2 < best_mbic2 - TOL)
        improved_k2   = (k < best_k_for_mbic2)
        if improved_val2 or improved_k2:
            if improved_val2:
                best_mbic2, best_sup_mbic2 = mbic2, supp.copy()
            best_k_for_mbic2 = min(best_k_for_mbic2, k)
            patience_mbic2 = 0
        else:
            patience_mbic2 += 1

        if patience_mbic >= NO_IMPROVE_PATIENCE and patience_mbic2 >= NO_IMPROVE_PATIENCE:
            break

    mBIC_fin  = float(best_mbic)  if np.isfinite(best_mbic)  else np.nan
    mBIC2_fin = float(best_mbic2) if np.isfinite(best_mbic2) else np.nan

    return ModelSelResult(
        mBIC=mBIC_fin,
        mBIC2=mBIC2_fin,
        model1=best_sup_mbic + 1,    # 1-basiert
        model2=best_sup_mbic2 + 1,
    )


# In[24]:


def lassonet_plus(
    y: np.ndarray,
    X: np.ndarray,
    model: str = "linear",

) -> ModelSelResult:
    """
    LassoNet mit identischen Hyperparametern wie `lassonet`, aber erweiterter
    mBIC-/mBIC2-gesteuerter Nachbearbeitung:
      - Top-K auf Basis feature_importances_ + Feinsuche
      - Großer Korrelations-Screen
      - Residual-Refinement (Add, Backward-Prune, Swap) für linear & logistic
    Gibt wie gehabt mBIC/mBIC2 und die beiden Modelle (1-basiert) zurück.
    """

    # ---- Parameter (IDENTISCH zu `lassonet`) ----
    NO_IMPROVE_PATIENCE = 4
    TOL = 1e-9
    SAT_DELTA = 5
    HIDDEN_DIMS = (4,)
    PATH_MULTIPLIER = 1.08
    N_ITERS = 100
    EARLY_PATIENCE = 2
    BATCH_SIZE = min(X.shape[0], 512)
    RANDOM_STATE = 42
    M = 30
    DEVICE = "cpu"

    # ---- Vorbereitungen ----
    model = model.lower().strip()
    if model not in ("linear", "logistic"):
        raise ValueError("lassonet_plus: model must be 'linear' or 'logistic'.")
    is_logistic = (model == "logistic")

    n, p = X.shape
    SAT_CUTOFF = min(int(p/2), int(n/2), 150)
    K_MAX = SAT_CUTOFF

    def _ll_full(supp: np.ndarray) -> tuple[float, int]:
        """Log-Likelihood für gegebenen Support (identisch zur Logik in `lassonet`)."""
        k = supp.size
        if k == 0:
            if is_logistic:
                p_hat = float(y.mean())
                p_hat = min(max(p_hat, 1e-12), 1 - 1e-12)
                ll = float((y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat)).sum())
            else:
                sigma2_0 = max(float(np.var(y, ddof=0)), 1e-12)
                ll = -0.5 * n * np.log(sigma2_0)
            return ll, 0

        Xk = X[:, supp]
        if is_logistic:
            mdl = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(Xk, y.astype(int))
            proba = np.clip(mdl.predict_proba(Xk)[:, 1], 1e-12, 1 - 1e-12)
            ll = float((y * np.log(proba) + (1 - y) * np.log(1 - proba)).sum())
        else:
            mdl = LinearRegression().fit(Xk, y)
            rss = float(((mdl.predict(Xk) - y) ** 2).sum())
            sigma2 = max(rss / n, 1e-12)
            ll = -0.5 * n * np.log(sigma2)
        return ll, k

    # ---- LassoNet Setup (identisch) ----
    net_cls = LassoNetClassifier if is_logistic else LassoNetRegressor
    net = net_cls(
        hidden_dims=HIDDEN_DIMS,
        path_multiplier=PATH_MULTIPLIER,
        lambda_start="auto",
        n_iters=N_ITERS,
        patience=EARLY_PATIENCE,
        batch_size=BATCH_SIZE,
        val_size=0.0,
        random_state=RANDOM_STATE,
        M=M,
        device=DEVICE,
        optim=(torch.optim.Adam, torch.optim.Adam),
    )

    X_net, y_net = _prepare_for_lassonet(X, y)
    X_path = X_net
    y_fit  = y_net
    #X_path = X * float(np.sqrt(n))
    #y_fit = y.astype(int) if is_logistic else y

    # ---- Pfad-Scan (identisch) ----
    best_mbic = float("inf");   best_sup_mbic = np.array([], dtype=int);   patience_mbic = 0
    best_mbic2 = float("inf");  best_sup_mbic2 = np.array([], dtype=int);  patience_mbic2 = 0
    best_k_for_mbic  = np.inf
    best_k_for_mbic2 = np.inf

    seen_supports: set[tuple[int, ...]] = set()
    last_state_dict = None

    for _, ckpt in enumerate(net.path(X_path, y_fit)):
        selected = getattr(ckpt, "selected", getattr(ckpt, "selected_features_", None))
        sd = getattr(ckpt, "state_dict", None)
        if sd is not None:
            last_state_dict = sd
        if selected is None:
            continue

        mask = np.asarray(selected, dtype=bool)
        df = int(mask.sum())
        if df >= SAT_CUTOFF:
            continue

        supp = np.flatnonzero(mask)
        key = tuple(supp.tolist())
        if key in seen_supports:
            continue
        seen_supports.add(key)

        ll, k = _ll_full(supp)
        mbic  = _mbic_bigstep(ll, n, k, p)
        mbic2 = _mbic2_bigstep(ll, n, k, p)

        improved_val = (mbic < best_mbic - TOL)
        improved_k   = (k < best_k_for_mbic)
        if improved_val or improved_k:
            if improved_val:
                best_mbic, best_sup_mbic = mbic, supp.copy()
            best_k_for_mbic = min(best_k_for_mbic, k)
            patience_mbic = 0
        else:
            patience_mbic += 1

        improved_val2 = (mbic2 < best_mbic2 - TOL)
        improved_k2   = (k < best_k_for_mbic2)
        if improved_val2 or improved_k2:
            if improved_val2:
                best_mbic2, best_sup_mbic2 = mbic2, supp.copy()
            best_k_for_mbic2 = min(best_k_for_mbic2, k)
            patience_mbic2 = 0
        else:
            patience_mbic2 += 1

        if patience_mbic >= NO_IMPROVE_PATIENCE and patience_mbic2 >= NO_IMPROVE_PATIENCE:
            break

    # ------------------------------
    # Top-K auf Basis feature_importances_ + Feinsuche
    # ------------------------------
    if last_state_dict is not None:
        try:
            net.load_state_dict(last_state_dict)
        except Exception:
            try:
                net.load(last_state_dict)
            except Exception:
                pass

    # robustes Ranking: fallback auf |X^Ty|
    try:
        imp_final  = np.asarray(net.feature_importances_)
        rank_theta = np.argsort(np.abs(imp_final))[::-1]
    except Exception:
        rank_theta = np.argsort(np.abs(X.T @ y))[::-1]

    K_hi = min(K_MAX, p)

    def _score_S(S: np.ndarray):
        ll, kk = _ll_full(S)
        return _mbic_bigstep(ll, n, kk, p), _mbic2_bigstep(ll, n, kk, p)

    def _score_K(K: int):
        S = rank_theta[:K]
        v1, v2 = _score_S(S)
        return v1, v2, S

    K_seed = int(best_sup_mbic.size) if best_sup_mbic.size > 0 else min(8, K_hi)
    WINDOW = 10
    STEP   = 2
    K_low  = max(1, K_seed - WINDOW)
    K_up   = min(K_seed + WINDOW, K_hi)

    best_theta_mBIC,  best_theta_sup,  K_best  = best_mbic,  best_sup_mbic.copy(),  K_seed
    best_theta_mBIC2, best_theta_sup2, K_best2 = best_mbic2, best_sup_mbic2.copy(), K_seed

    for K in range(K_low, K_up + 1, STEP):
        v1, v2, S = _score_K(K)
        if v1 < best_theta_mBIC - TOL:
            best_theta_mBIC, best_theta_sup,  K_best  = v1, S.copy(), K
        if v2 < best_theta_mBIC2 - TOL:
            best_theta_mBIC2, best_theta_sup2, K_best2 = v2, S.copy(), K

    for K in range(max(1, K_best  - 3), min(K_hi, K_best  + 3) + 1):
        v1, v2, S = _score_K(K)
        if v1 < best_theta_mBIC - TOL:
            best_theta_mBIC, best_theta_sup = v1, S.copy()

    for K in range(max(1, K_best2 - 3), min(K_hi, K_best2 + 3) + 1):
        v1, v2, S = _score_K(K)
        if v2 < best_theta_mBIC2 - TOL:
            best_theta_mBIC2, best_theta_sup2 = v2, S.copy()

    if best_theta_mBIC  < best_mbic  - TOL: best_mbic,  best_sup_mbic  = best_theta_mBIC,  best_theta_sup
    if best_theta_mBIC2 < best_mbic2 - TOL: best_mbic2, best_sup_mbic2 = best_theta_mBIC2, best_theta_sup2


    # ------------------------------
    # Großer Korrelations-Screen 
    # ------------------------------
    K_screen = min(K_MAX, 70)
    scores_glob = np.abs(X.T @ y)
    ranked = np.argsort(scores_glob)[::-1][:K_screen]

    Ks_scr = [1]
    while Ks_scr[-1] < K_screen:
        Ks_scr.append(min(K_screen, Ks_scr[-1] * 2))
    if best_sup_mbic.size > 0:
        Ks_scr.append(min(K_screen, best_sup_mbic.size))
    Ks_scr = sorted(set(Ks_scr))

    best_scr_mBIC,  best_scr_sup  = best_mbic,  best_sup_mbic
    best_scr_mBIC2, best_scr_sup2 = best_mbic2, best_sup_mbic2

    for K in Ks_scr:
        S = ranked[:K]
        v1, v2 = _score_S(S)
        if v1 < best_scr_mBIC - TOL:
            best_scr_mBIC, best_scr_sup = v1, S.copy()
        if v2 < best_scr_mBIC2 - TOL:
            best_scr_mBIC2, best_scr_sup2 = v2, S.copy()

    if best_scr_mBIC < best_mbic - TOL:
        best_mbic, best_sup_mbic = best_scr_mBIC, best_scr_sup
    if best_scr_mBIC2 < best_mbic2 - TOL:
        best_mbic2, best_sup_mbic2 = best_scr_mBIC2, best_scr_sup2

    # ------------------------------
    # Residual-Refinement (linear & logistic) – getrennt für mBIC und mBIC2
    # ------------------------------
    def _fit_and_residuals(S: np.ndarray):
        if S.size == 0:
            if is_logistic:
                p0 = float(np.clip(y.mean(), 1e-12, 1-1e-12))
                r  = y - p0
                return None, r, None
            else:
                return None, y, None
        Xs = X[:, S]
        if is_logistic:
            mdl = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(Xs, y.astype(int))
            proba = np.clip(mdl.predict_proba(Xs)[:, 1], 1e-12, 1 - 1e-12)
            r = y - proba
            coef = mdl.coef_.ravel()
            return mdl, r, coef
        else:
            mdl = LinearRegression().fit(Xs, y)
            r = y - mdl.predict(Xs)
            coef = mdl.coef_
            return mdl, r, coef

    if (best_sup_mbic.size < K_MAX) or (best_sup_mbic2.size < K_MAX):
        max_local_steps = 6
        cand_cap_add    = min(3 * K_MAX, 240)
        cand_cap_swap   = min(2 * K_MAX, 180)

        # ===========================
        # PASS 1: mBIC verfeinern
        # ===========================
        if best_sup_mbic.size < K_MAX:
            # (0) Backward-Prune (einmalig)
            improved = True
            while improved and best_sup_mbic.size > 0:
                improved = False
                S = best_sup_mbic
                for i in range(S.size):
                    S_try = np.delete(S, i)
                    v_try, _ = _score_S(S_try)  # (mBIC, mBIC2)
                    if v_try < best_mbic - TOL:
                        best_mbic, best_sup_mbic = v_try, S_try
                        improved = True
                        break  # Neustart der Schleife mit neuem Support

            # (A) Greedy Add (bis keine Verbesserung)
            for _ in range(max_local_steps):
                _, r, _ = _fit_and_residuals(best_sup_mbic)
                scores_r = np.abs(X.T @ r)
                S_set = set(best_sup_mbic.tolist())
                cand = [j for j in np.argsort(scores_r)[::-1] if j not in S_set][:cand_cap_add]
                improved = False
                for j in cand:
                    S_try = np.array(sorted(S_set | {j}), dtype=int)
                    if S_try.size > K_MAX:
                        continue
                    v_try, _ = _score_S(S_try)
                    if v_try < best_mbic - TOL:
                        best_mbic, best_sup_mbic = v_try, S_try
                        improved = True
                        break
                if not improved:
                    break

            # (B) Swap (klein gehalten)
            mdl, r, beta = _fit_and_residuals(best_sup_mbic)
            if beta is not None and beta.size > 0:
                order_in = np.argsort(np.abs(beta))  # schwächste zuerst
                scores_r = np.abs(X.T @ r)
                S_set    = set(best_sup_mbic.tolist())
                cand_out = order_in[: min(10, beta.size)]
                cand_in  = [j for j in np.argsort(scores_r)[::-1] if j not in S_set][:cand_cap_swap]

                improved = True
                iter_swap = 0
                while improved and iter_swap < 4:
                    improved = False
                    for idx_worst in cand_out:
                        for j in cand_in:
                            if j in S_set:
                                continue
                            S_try = best_sup_mbic.copy()
                            S_try[idx_worst] = j
                            S_try = np.array(sorted(set(S_try.tolist())), dtype=int)
                            if S_try.size > K_MAX:
                                continue
                            v_try, _ = _score_S(S_try)
                            if v_try < best_mbic - TOL:
                                best_mbic, best_sup_mbic = v_try, S_try
                                S_set = set(S_try.tolist())
                                improved = True
                                break
                        if improved:
                            break
                    iter_swap += 1

        # ===========================
        # PASS 2: mBIC2 verfeinern
        # ===========================
        if best_sup_mbic2.size < K_MAX:
            # (0) Backward-Prune (einmalig)
            improved = True
            while improved and best_sup_mbic2.size > 0:
                improved = False
                S2 = best_sup_mbic2
                for i in range(S2.size):
                    S2_try = np.delete(S2, i)
                    _, v2_try = _score_S(S2_try)  # (mBIC, mBIC2)
                    if v2_try < best_mbic2 - TOL:
                        best_mbic2, best_sup_mbic2 = v2_try, S2_try
                        improved = True
                        break

            # (A) Greedy Add (bis keine Verbesserung)
            for _ in range(max_local_steps):
                _, r2, _ = _fit_and_residuals(best_sup_mbic2)
                scores_r2 = np.abs(X.T @ r2)
                S2_set = set(best_sup_mbic2.tolist())
                cand2 = [j for j in np.argsort(scores_r2)[::-1] if j not in S2_set][:cand_cap_add]
                improved = False
                for j in cand2:
                    S2_try = np.array(sorted(S2_set | {j}), dtype=int)
                    if S2_try.size > K_MAX:
                        continue
                    _, v2_try = _score_S(S2_try)
                    if v2_try < best_mbic2 - TOL:
                        best_mbic2, best_sup_mbic2 = v2_try, S2_try
                        improved = True
                        break
                if not improved:
                    break

            # (B) Swap (klein gehalten)
            mdl2, r2, beta2 = _fit_and_residuals(best_sup_mbic2)
            if beta2 is not None and beta2.size > 0:
                order_in2 = np.argsort(np.abs(beta2))  # schwächste zuerst
                scores_r2 = np.abs(X.T @ r2)
                S2_set    = set(best_sup_mbic2.tolist())
                cand_out2 = order_in2[: min(10, beta2.size)]
                cand_in2  = [j for j in np.argsort(scores_r2)[::-1] if j not in S2_set][:cand_cap_swap]

                improved = True
                iter_swap2 = 0
                while improved and iter_swap2 < 4:
                    improved = False
                    for idx_worst2 in cand_out2:
                        for j in cand_in2:
                            if j in S2_set:
                                continue
                            S2_try = best_sup_mbic2.copy()
                            S2_try[idx_worst2] = j
                            S2_try = np.array(sorted(set(S2_try.tolist())), dtype=int)
                            if S2_try.size > K_MAX:
                                continue
                            _, v2_try = _score_S(S2_try)
                            if v2_try < best_mbic2 - TOL:
                                best_mbic2, best_sup_mbic2 = v2_try, S2_try
                                S2_set = set(S2_try.tolist())
                                improved = True
                                break
                        if improved:
                            break
                    iter_swap2 += 1


    # Finale Werte
    mBIC_fin  = float(best_mbic)  if np.isfinite(best_mbic)  else np.nan
    mBIC2_fin = float(best_mbic2) if np.isfinite(best_mbic2) else np.nan

    return ModelSelResult(
        mBIC=mBIC_fin,
        mBIC2=mBIC2_fin,
        model1=best_sup_mbic + 1,   # 1-basiert
        model2=best_sup_mbic2 + 1,
    )


# In[ ]:


# ---------- Hilfsfunktionen --------------------------------
def _l21_norm(w: torch.Tensor) -> torch.Tensor:
    """Zeilen‑weise L2‑Norm, dann Summe (||·||₂,₁)."""
    return torch.norm(w, dim=0).sum()


# In[25]:


def deep2stage(
    y: np.ndarray,
    X: np.ndarray,
    model: str = "linear",          # "linear" | "logistic"
    h: int = 128,                    # Hidden-Dim der Autoencoder-Bottleneck-Schicht
    stage1_epochs: int = 250,
    stage2_epochs: int = 250,
    batch_size: int = 512,
    λ: float = 0.1,                 # Rekonstruktionsgewicht (Stage 1)
    α: float | None = None,         # L2,1-Penalty (Stage 2)
    β: float = 0,                   # Frobenius-Decay (Stage 2)
    lr: float = 1e-3,
    K_max: int | None = None,       # Obergrenze Support-Größe für mBIC-Suche
    patience: int = 12,
    device: str | torch.device = "cpu",
) -> ModelSelResult:
    """
    Zweiseitiges DNN-Feature-Screening ohne Validierungsset.
    Early-Stopping basiert auf dem (gemittelten) Train-Loss je Epoche.
    """
    model = model.lower().strip()
    is_logistic = (model == "logistic")
    if α is None:
        α = 1e-7 if is_logistic else 1e-4
    if model not in ("linear", "logistic"):
        raise ValueError("deep2stage: model muss 'linear' oder 'logistic' sein.")

    n, p = X.shape
    if K_max is None:
        K_MAX = min(int(0.5*n), int(0.5*p), 100)
    else:
        K_MAX = int(K_max)

    # ----- Tensors (gesamter Datensatz als 'Train') -----
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    if is_logistic:
        y_int = y.astype(int)
        y_t = torch.tensor(y_int, dtype=torch.long, device=device)
    else:
        y_t = torch.tensor(y, dtype=torch.float32, device=device)

    # -------------- Stage 1 – (Supervised) Autoencoder -----
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
            self.head = nn.Linear(h, 2 if is_logistic else 1)

        def forward(self, x):
            z = self.enc(x)
            x_hat = self.dec(z)
            y_hat = self.head(z)
            return z, x_hat, y_hat

    sae = SAE().to(device)
    opt1 = optim.RMSprop(sae.parameters(), lr=lr)

    ds = DataLoader(TensorDataset(X_t, y_t),
                    batch_size=batch_size, shuffle=True)

    best_state1, best_loss1, no_imp1 = None, float("inf"), 0
    for _ in range(stage1_epochs):
        sae.train()
        running, nb = 0.0, 0
        for xb, yb in ds:
            opt1.zero_grad()
            z, x_hat, y_hat = sae(xb)
            recon = nn.functional.mse_loss(x_hat, xb)
            if is_logistic:
                sup = nn.functional.cross_entropy(y_hat, yb)
            else:
                sup = nn.functional.mse_loss(y_hat.squeeze(), yb)
            loss = sup + λ * recon
            loss.backward()
            opt1.step()
            running += float(loss.item())
            nb += 1
        epoch_loss = running / max(nb, 1)
        if epoch_loss < best_loss1 - 1e-6:
            best_loss1, no_imp1 = epoch_loss, 0
            best_state1 = {k: v.detach().clone() for k, v in sae.state_dict().items()}
        else:
            no_imp1 += 1
            if no_imp1 >= patience:
                break  # Early-Stopping Stage 1 (Train-Loss)

    if best_state1 is not None:
        sae.load_state_dict(best_state1)

    # Bottleneck-Features für alle Beobachtungen + Min-Max-Norm
    with torch.no_grad():
        x_encode = sae.enc(X_t)
        x_encode = (x_encode - x_encode.min(0).values) / \
                   (x_encode.max(0).values - x_encode.min(0).values + 1e-8)

    # -------------- Stage 2 – Regularisiertes 1-Hidden-Net --
    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(p, h)
            self.out = nn.Linear(h, h, bias=True)
        def forward(self, x):
            return self.out(torch.relu(self.fc1(x)))

    stu = Student().to(device)
    opt2 = optim.RMSprop(stu.parameters(), lr=lr)

    best_state2, best_loss2, no_imp2 = None, float("inf"), 0
    for _ in range(stage2_epochs):
        stu.train()
        opt2.zero_grad()
        x_hat_all = stu(X_t)
        mse = nn.functional.mse_loss(x_hat_all, x_encode)
        l21 = _l21_norm(stu.fc1.weight)
        frob = stu.fc1.weight.norm()**2 + stu.out.weight.norm()**2
        loss2 = mse + α * l21 + (β / 2) * frob
        loss2.backward()
        opt2.step()

        train_loss = float(loss2.item())
        if train_loss < best_loss2 - 1e-6:
            best_loss2, no_imp2 = train_loss, 0
            best_state2 = {k: v.detach().clone() for k, v in stu.state_dict().items()}
        else:
            no_imp2 += 1
            if no_imp2 >= patience:
                break  # Early-Stopping Stage 2 (Train-Loss)

    if best_state2 is not None:
        stu.load_state_dict(best_state2)

    # --------------------------------------------------------
    # Feature-Scores → Ranking
    # --------------------------------------------------------
    with torch.no_grad():
        W1 = stu.fc1.weight                                  # h × p
        scores = torch.sum(W1 * W1, dim=0).cpu().numpy()     # diag(W₁ᵀW₁)
    ranked = scores.argsort()[::-1]

    # --------------------------------------------------------
    # mBIC / mBIC2 entlang der Top-k-Supports
    # --------------------------------------------------------
    best_BIC  = best_BIC2 = float("inf")
    best_sup1 = best_sup2 = np.array([], dtype=int)

    n_float = float(n)
    for k in range(0, min(K_MAX, p) + 1):
        supp = ranked[:k]

        if k == 0:
            # ----- Log-Likelihood ohne Prädiktoren ---------------
            if is_logistic:
                eps = 1e-12
                p_hat = np.clip(y_int.mean(), eps, 1 - eps)
                ll = float((y_int * np.log(p_hat) + (1 - y_int) * np.log(1 - p_hat)).sum())
            else:
                sigma2_0 = np.var(y, ddof=0)
                ll = float(-0.5 * n_float * np.log(sigma2_0 + 1e-12))
            df = 0
        else:
            Xk = X[:, supp]
            if is_logistic:
                try:
                    mdl = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(Xk, y_int)
                except Exception:
                    # Fallback für ältere sklearn-Versionen
                    mdl = LogisticRegression(penalty="l2", C=1e12, solver="lbfgs", max_iter=2000).fit(Xk, y_int)
                # LL via proba + Clipping (deine Variante)
                class1_col = int(np.where(mdl.classes_ == 1)[0][0])
                proba = mdl.predict_proba(Xk)[:, class1_col]
                proba = np.clip(proba, 1e-12, 1 - 1e-12)
                ll = float((y_int * np.log(proba) + (1 - y_int) * np.log(1 - proba)).sum())
            else:
                mdl = LinearRegression().fit(Xk, y)
                rss = float(((mdl.predict(Xk) - y) ** 2).sum())
                sigma2 = rss / n_float
                ll = float(-0.5 * n_float * np.log(sigma2 + 1e-12))
            df = k

        mBIC  = _mbic_bigstep(ll, n, k, p)
        mBIC2 = _mbic2_bigstep(ll, n, k, p)

        if mBIC < best_BIC:
            best_BIC, best_sup1 = mBIC, supp.copy()
        if mBIC2 < best_BIC2:
            best_BIC2, best_sup2 = mBIC2, supp.copy()

    # --------------------------------------------------------
    # Rückgabe (1-basiert)
    # --------------------------------------------------------
    return ModelSelResult(
        mBIC  = float(best_BIC),
        mBIC2 = float(best_BIC2),
        model1 = best_sup1 + 1,
        model2 = best_sup2 + 1
    )


# In[26]:


def deep2stage_plus(
    y: np.ndarray,
    X: np.ndarray,
    model: str = "linear",          # "linear" | "logistic"
    h: int = 128,                    # Hidden-Dim der Autoencoder-Bottleneck-Schicht
    stage1_epochs: int = 250,
    stage2_epochs: int = 250,
    batch_size: int = 512,
    λ: float = 0.1,                 # Rekonstruktionsgewicht (Stage 1)
    α: float | None = None,         # L2,1-Penalty (Stage 2)
    β: float = 0,                   # Frobenius-Decay (Stage 2)
    lr: float = 1e-3,
    K_max: int | None = None,       # Obergrenze Support-Größe für mBIC-Suche
    patience: int = 12,
    device: str | torch.device = "cpu",
) -> ModelSelResult:
    """
    deep2stage + (Top-K Feinsuche, Korrelations-Screen, Residual-Refinement).
    Early-Stopping bleibt auf Train-Loss je Epoche.
    """
    model = model.lower().strip()
    is_logistic = (model == "logistic")
    if α is None:
        α = 1e-7 if is_logistic else 1e-4
    if model not in ("linear", "logistic"):
        raise ValueError("deep2stage_plus: model muss 'linear' oder 'logistic' sein.")

    n, p = X.shape
    if K_max is None:
        K_MAX = min(int(0.5*n), int(0.5*p), 100)
    else:
        K_MAX = int(K_max)
    K_hi = min(K_MAX, p)
    TOL = 1e-9

    # ----- Tensors -----
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    if is_logistic:
        y_int = y.astype(int)
        y_t = torch.tensor(y_int, dtype=torch.long, device=device)
    else:
        y_t = torch.tensor(y, dtype=torch.float32, device=device)

    # -------------- Stage 1 – (Supervised) Autoencoder -----
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
            self.head = nn.Linear(h, 2 if is_logistic else 1)

        def forward(self, x):
            z = self.enc(x)
            x_hat = self.dec(z)
            y_hat = self.head(z)
            return z, x_hat, y_hat

    sae = SAE().to(device)
    opt1 = optim.RMSprop(sae.parameters(), lr=lr)
    ds = DataLoader(TensorDataset(X_t, y_t),
                    batch_size=batch_size, shuffle=True)

    best_state1, best_loss1, no_imp1 = None, float("inf"), 0
    for _ in range(stage1_epochs):
        sae.train()
        running, nb = 0.0, 0
        for xb, yb in ds:
            opt1.zero_grad()
            z, x_hat, y_hat = sae(xb)
            recon = nn.functional.mse_loss(x_hat, xb)
            if is_logistic:
                sup = nn.functional.cross_entropy(y_hat, yb)
            else:
                sup = nn.functional.mse_loss(y_hat.squeeze(), yb)
            loss = sup + λ * recon
            loss.backward()
            opt1.step()
            running += float(loss.item()); nb += 1
        epoch_loss = running / max(nb, 1)
        if epoch_loss < best_loss1 - 1e-6:
            best_loss1, no_imp1 = epoch_loss, 0
            best_state1 = {k: v.detach().clone() for k, v in sae.state_dict().items()}
        else:
            no_imp1 += 1
            if no_imp1 >= patience:
                break

    if best_state1 is not None:
        sae.load_state_dict(best_state1)

    # Bottleneck-Features (Min-Max-Norm)
    with torch.no_grad():
        x_encode = sae.enc(X_t)
        x_encode = (x_encode - x_encode.min(0).values) / \
                   (x_encode.max(0).values - x_encode.min(0).values + 1e-8)

    # -------------- Stage 2 – Regularisiertes 1-Hidden-Net --
    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(p, h)
            self.out = nn.Linear(h, h, bias=True)
        def forward(self, x):
            return self.out(torch.relu(self.fc1(x)))

    stu = Student().to(device)
    opt2 = optim.RMSprop(stu.parameters(), lr=lr)

    best_state2, best_loss2, no_imp2 = None, float("inf"), 0
    for _ in range(stage2_epochs):
        stu.train()
        opt2.zero_grad()
        x_hat_all = stu(X_t)
        mse = nn.functional.mse_loss(x_hat_all, x_encode)
        l21 = _l21_norm(stu.fc1.weight)
        frob = stu.fc1.weight.norm()**2 + stu.out.weight.norm()**2
        loss2 = mse + α * l21 + (β / 2) * frob
        loss2.backward()
        opt2.step()

        train_loss = float(loss2.item())
        if train_loss < best_loss2 - 1e-6:
            best_loss2, no_imp2 = train_loss, 0
            best_state2 = {k: v.detach().clone() for k, v in stu.state_dict().items()}
        else:
            no_imp2 += 1
            if no_imp2 >= patience:
                break

    if best_state2 is not None:
        stu.load_state_dict(best_state2)

    # --------------------------------------------------------
    # Feature-Scores → Ranking (Importances)
    # --------------------------------------------------------
    with torch.no_grad():
        W1 = stu.fc1.weight                                  # h × p
        scores = torch.sum(W1 * W1, dim=0).cpu().numpy()     # diag(W₁ᵀW₁)
    ranked = scores.argsort()[::-1]

    # --------------------------------------------------------
    # (Baseline) mBIC / mBIC2 entlang der Top-k-Supports
    # --------------------------------------------------------
    def _ll_full(supp: np.ndarray) -> tuple[float, int]:
        k = supp.size
        if k == 0:
            if is_logistic:
                eps = 1e-12
                p_hat = float(np.clip(y.mean(), eps, 1 - eps))
                ll = float((y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat)).sum())
            else:
                sigma2_0 = max(float(np.var(y, ddof=0)), 1e-12)
                ll = -0.5 * n * np.log(sigma2_0)
            return ll, 0
        Xk = X[:, supp]
        if is_logistic:
            try:
                mdl = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(Xk, y.astype(int))
            except Exception:
                mdl = LogisticRegression(penalty="l2", C=1e12, solver="lbfgs", max_iter=2000).fit(Xk, y.astype(int))
            proba = np.clip(mdl.predict_proba(Xk)[:, 1], 1e-12, 1 - 1e-12)
            ll = float((y * np.log(proba) + (1 - y) * np.log(1 - proba)).sum())
        else:
            mdl = LinearRegression().fit(Xk, y)
            rss = float(((mdl.predict(Xk) - y) ** 2).sum())
            sigma2 = max(rss / n, 1e-12)
            ll = -0.5 * n * np.log(sigma2)
        return ll, k

    best_mbic  = float("inf"); best_sup_mbic  = np.array([], dtype=int)
    best_mbic2 = float("inf"); best_sup_mbic2 = np.array([], dtype=int)

    for k in range(0, min(K_MAX, p) + 1):
        supp = ranked[:k]
        ll, df = _ll_full(supp)
        mbic  = _mbic_bigstep(ll, n, df, p)
        mbic2 = _mbic2_bigstep(ll, n, df, p)
        if mbic < best_mbic:
            best_mbic, best_sup_mbic = mbic, supp.copy()
        if mbic2 < best_mbic2:
            best_mbic2, best_sup_mbic2 = mbic2, supp.copy()

    # ========================================================
    # Top-K auf Basis Importances + Feinsuche
    # (feature_importances_ → hier: scores aus W1; Fallback: |X^Ty|)
    # ========================================================
    try:
        imp_final = np.asarray(scores)
        rank_theta = np.argsort(np.abs(imp_final))[::-1]
    except Exception:
        rank_theta = np.argsort(np.abs(X.T @ (y if not is_logistic else y.astype(float))))[::-1]

    def _score_S(S: np.ndarray):
        ll, kk = _ll_full(S)
        return _mbic_bigstep(ll, n, kk, p), _mbic2_bigstep(ll, n, kk, p)

    def _score_K(K: int):
        S = rank_theta[:K]
        v1, v2 = _score_S(S)
        return v1, v2, S

    K_seed = int(best_sup_mbic.size) if best_sup_mbic.size > 0 else min(8, K_hi)
    WINDOW, STEP = 10, 2
    K_low  = max(1, K_seed - WINDOW)
    K_up   = min(K_seed + WINDOW, K_hi)

    best_theta_mBIC,  best_theta_sup,  K_best  = best_mbic,  best_sup_mbic.copy(),  K_seed
    best_theta_mBIC2, best_theta_sup2, K_best2 = best_mbic2, best_sup_mbic2.copy(), K_seed

    for K in range(K_low, K_up + 1, STEP):
        v1, v2, S = _score_K(K)
        if v1 < best_theta_mBIC - TOL:
            best_theta_mBIC, best_theta_sup,  K_best  = v1, S.copy(), K
        if v2 < best_theta_mBIC2 - TOL:
            best_theta_mBIC2, best_theta_sup2, K_best2 = v2, S.copy(), K

    for K in range(max(1, K_best  - 3), min(K_hi, K_best  + 3) + 1):
        v1, v2, S = _score_K(K)
        if v1 < best_theta_mBIC - TOL:
            best_theta_mBIC, best_theta_sup = v1, S.copy()

    for K in range(max(1, K_best2 - 3), min(K_hi, K_best2 + 3) + 1):
        v1, v2, S = _score_K(K)
        if v2 < best_theta_mBIC2 - TOL:
            best_theta_mBIC2, best_theta_sup2 = v2, S.copy()

    # getrennt zurückspielen
    if best_theta_mBIC  < best_mbic  - TOL: best_mbic,  best_sup_mbic  = best_theta_mBIC,  best_theta_sup
    if best_theta_mBIC2 < best_mbic2 - TOL: best_mbic2, best_sup_mbic2 = best_theta_mBIC2, best_theta_sup2

    # ========================================================
    # Großer Korrelations-Screen
    # ========================================================
    K_screen = min(K_MAX, 70)
    scores_glob = np.abs(X.T @ (y if not is_logistic else y.astype(float)))
    ranked_scr = np.argsort(scores_glob)[::-1][:K_screen]

    Ks_scr = [1]
    while Ks_scr[-1] < K_screen:
        Ks_scr.append(min(K_screen, Ks_scr[-1] * 2))
    if best_sup_mbic.size > 0:
        Ks_scr.append(min(K_screen, best_sup_mbic.size))
    Ks_scr = sorted(set(Ks_scr))

    best_scr_mBIC,  best_scr_sup  = best_mbic,  best_sup_mbic
    best_scr_mBIC2, best_scr_sup2 = best_mbic2, best_sup_mbic2

    for K in Ks_scr:
        S = ranked_scr[:K]
        v1, v2 = _score_S(S)
        if v1 < best_scr_mBIC - TOL:
            best_scr_mBIC, best_scr_sup = v1, S.copy()
        if v2 < best_scr_mBIC2 - TOL:
            best_scr_mBIC2, best_scr_sup2 = v2, S.copy()

    if best_scr_mBIC < best_mbic - TOL:
        best_mbic, best_sup_mbic = best_scr_mBIC, best_scr_sup
    if best_scr_mBIC2 < best_mbic2 - TOL:
        best_mbic2, best_sup_mbic2 = best_scr_mBIC2, best_scr_sup2

    # ------------------------------
    # Residual-Refinement (linear & logistic) – getrennt für mBIC und mBIC2
    # ------------------------------
    def _fit_and_residuals(S: np.ndarray):
        if S.size == 0:
            if is_logistic:
                p0 = float(np.clip(y.mean(), 1e-12, 1-1e-12))
                r  = y - p0
                return None, r, None
            else:
                return None, y, None
        Xs = X[:, S]
        if is_logistic:
            mdl = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(Xs, y.astype(int))
            proba = np.clip(mdl.predict_proba(Xs)[:, 1], 1e-12, 1 - 1e-12)
            r = y - proba
            coef = mdl.coef_.ravel()
            return mdl, r, coef
        else:
            mdl = LinearRegression().fit(Xs, y)
            r = y - mdl.predict(Xs)
            coef = mdl.coef_
            return mdl, r, coef

    if (best_sup_mbic.size < K_MAX) or (best_sup_mbic2.size < K_MAX):
        max_local_steps = 6
        cand_cap_add    = min(3 * K_MAX, 240)
        cand_cap_swap   = min(2 * K_MAX, 180)

        # ===========================
        # PASS 1: mBIC verfeinern
        # ===========================
        if best_sup_mbic.size < K_MAX:
            # (0) Backward-Prune (einmalig)
            improved = True
            while improved and best_sup_mbic.size > 0:
                improved = False
                S = best_sup_mbic
                for i in range(S.size):
                    S_try = np.delete(S, i)
                    v_try, _ = _score_S(S_try)  # (mBIC, mBIC2)
                    if v_try < best_mbic - TOL:
                        best_mbic, best_sup_mbic = v_try, S_try
                        improved = True
                        break  # Neustart der Schleife mit neuem Support

            # (A) Greedy Add (bis keine Verbesserung)
            for _ in range(max_local_steps):
                _, r, _ = _fit_and_residuals(best_sup_mbic)
                scores_r = np.abs(X.T @ r)
                S_set = set(best_sup_mbic.tolist())
                cand = [j for j in np.argsort(scores_r)[::-1] if j not in S_set][:cand_cap_add]
                improved = False
                for j in cand:
                    S_try = np.array(sorted(S_set | {j}), dtype=int)
                    if S_try.size > K_MAX:
                        continue
                    v_try, _ = _score_S(S_try)
                    if v_try < best_mbic - TOL:
                        best_mbic, best_sup_mbic = v_try, S_try
                        improved = True
                        break
                if not improved:
                    break

            # (B) Swap (klein gehalten)
            mdl, r, beta = _fit_and_residuals(best_sup_mbic)
            if beta is not None and beta.size > 0:
                order_in = np.argsort(np.abs(beta))  # schwächste zuerst
                scores_r = np.abs(X.T @ r)
                S_set    = set(best_sup_mbic.tolist())
                cand_out = order_in[: min(10, beta.size)]
                cand_in  = [j for j in np.argsort(scores_r)[::-1] if j not in S_set][:cand_cap_swap]

                improved = True
                iter_swap = 0
                while improved and iter_swap < 4:
                    improved = False
                    for idx_worst in cand_out:
                        for j in cand_in:
                            if j in S_set:
                                continue
                            S_try = best_sup_mbic.copy()
                            S_try[idx_worst] = j
                            S_try = np.array(sorted(set(S_try.tolist())), dtype=int)
                            if S_try.size > K_MAX:
                                continue
                            v_try, _ = _score_S(S_try)
                            if v_try < best_mbic - TOL:
                                best_mbic, best_sup_mbic = v_try, S_try
                                S_set = set(S_try.tolist())
                                improved = True
                                break
                        if improved:
                            break
                    iter_swap += 1

        # ===========================
        # PASS 2: mBIC2 verfeinern
        # ===========================
        if best_sup_mbic2.size < K_MAX:
            # (0) Backward-Prune (einmalig)
            improved = True
            while improved and best_sup_mbic2.size > 0:
                improved = False
                S2 = best_sup_mbic2
                for i in range(S2.size):
                    S2_try = np.delete(S2, i)
                    _, v2_try = _score_S(S2_try)  # (mBIC, mBIC2)
                    if v2_try < best_mbic2 - TOL:
                        best_mbic2, best_sup_mbic2 = v2_try, S2_try
                        improved = True
                        break

            # (A) Greedy Add (bis keine Verbesserung)
            for _ in range(max_local_steps):
                _, r2, _ = _fit_and_residuals(best_sup_mbic2)
                scores_r2 = np.abs(X.T @ r2)
                S2_set = set(best_sup_mbic2.tolist())
                cand2 = [j for j in np.argsort(scores_r2)[::-1] if j not in S2_set][:cand_cap_add]
                improved = False
                for j in cand2:
                    S2_try = np.array(sorted(S2_set | {j}), dtype=int)
                    if S2_try.size > K_MAX:
                        continue
                    _, v2_try = _score_S(S2_try)
                    if v2_try < best_mbic2 - TOL:
                        best_mbic2, best_sup_mbic2 = v2_try, S2_try
                        improved = True
                        break
                if not improved:
                    break

            # (B) Swap (klein gehalten)
            mdl2, r2, beta2 = _fit_and_residuals(best_sup_mbic2)
            if beta2 is not None and beta2.size > 0:
                order_in2 = np.argsort(np.abs(beta2))  # schwächste zuerst
                scores_r2 = np.abs(X.T @ r2)
                S2_set    = set(best_sup_mbic2.tolist())
                cand_out2 = order_in2[: min(10, beta2.size)]
                cand_in2  = [j for j in np.argsort(scores_r2)[::-1] if j not in S2_set][:cand_cap_swap]

                improved = True
                iter_swap2 = 0
                while improved and iter_swap2 < 4:
                    improved = False
                    for idx_worst2 in cand_out2:
                        for j in cand_in2:
                            if j in S2_set:
                                continue
                            S2_try = best_sup_mbic2.copy()
                            S2_try[idx_worst2] = j
                            S2_try = np.array(sorted(set(S2_try.tolist())), dtype=int)
                            if S2_try.size > K_MAX:
                                continue
                            _, v2_try = _score_S(S2_try)
                            if v2_try < best_mbic2 - TOL:
                                best_mbic2, best_sup_mbic2 = v2_try, S2_try
                                S2_set = set(S2_try.tolist())
                                improved = True
                                break
                        if improved:
                            break
                    iter_swap2 += 1


    # --------------------------------------------------------
    # Rückgabe (1-basiert)
    # --------------------------------------------------------
    mBIC_fin  = float(best_mbic)  if np.isfinite(best_mbic)  else np.nan
    mBIC2_fin = float(best_mbic2) if np.isfinite(best_mbic2) else np.nan

    return ModelSelResult(
        mBIC  = mBIC_fin,
        mBIC2 = mBIC2_fin,
        model1 = best_sup_mbic + 1,
        model2 = best_sup_mbic2 + 1
    )


# In[27]:


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
    "lassonet_plus",
    "deep2stage", 
    "deep2stage_plus",
]


# In[ ]:





# In[ ]:




