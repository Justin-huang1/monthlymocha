# bayes_core.py
# Minimal deps: numpy, pandas. (Optional: scipy for F-test p-values)

import numpy as np
import pandas as pd
import itertools

# ============== IO & FEATURE BUILDERS ==============

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df

def detect_channels(df: pd.DataFrame):
    spend_cols = [c for c in df.columns if c.endswith("_spend")]
    channels = [c.replace("_spend","") for c in spend_cols]
    imp_cols = [f"{c}_impressions" for c in channels if f"{c}_impressions" in df.columns]
    return channels, spend_cols, imp_cols

def hill_saturation(x, alpha=1.0, half_saturation=1.0):
    x = np.asarray(x, dtype=float)
    return alpha * x / (x + half_saturation)

def adstock_geometric(x, decay=0.5, max_lag=8):
    x = np.asarray(x, dtype=float)
    n = len(x)
    w = np.array([decay**k for k in range(max_lag+1)], dtype=float)
    out = np.zeros(n)
    for t in range(n):
        kmax = min(max_lag, t)
        out[t] = (x[t-kmax:t+1][::-1] * w[:kmax+1]).sum()
    return out

def build_features(df: pd.DataFrame, decay=0.5, max_lag=8, sat_pct=0.6):
    channels, spend_cols, _ = detect_channels(df)
    X_list, names = [], []
    for ch in channels:
        xs = df[f"{ch}_spend"].astype(float).values
        xa = adstock_geometric(xs, decay=decay, max_lag=max_lag)
        X_list.append(xa); names.append(f"{ch}_spend_adstock")
        impc = f"{ch}_impressions"
        if impc in df.columns:
            xi = df[impc].astype(float).values
            hs = np.percentile(xi[xi > 0], 100*sat_pct) if (xi > 0).any() else 1.0
            xsat = hill_saturation(xi, alpha=1.0, half_saturation=hs)
            X_list.append(xsat); names.append(f"{ch}_impr_sat")
    X = np.column_stack(X_list) if X_list else np.empty((len(df), 0))
    y = df["subscriptions"].astype(float).values
    return X, y, names

# ============== BMA (hyper-g) & g-PRIOR HELPERS ==============


def _rss_and_r2_stable(X, y):
    # solves with lstsq (SVD), works when XtX is singular
    beta_hat, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_hat
    rss = float(np.dot(resid, resid))
    tss = float(((y - y.mean())**2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    return rss, r2, beta_hat, rank

def log_marginal_hyper_g(X, y, a=3.0, grid=None):
    # X should already include intercept if you want one
    n, p = X.shape
    _, r2, _, rank = _rss_and_r2_stable(X, y)
    if rank < min(n, p):  # rank deficient -> near singular
        # Penalize this model heavily so it's effectively ignored
        return -1e12

    if grid is None:
        grid = np.logspace(-2, 4, 64)

    li = []
    for g in grid:
        li.append(-0.5*p*np.log(1.0+g) - 0.5*(n-1)*np.log(1.0 + g*(1.0-r2)) - (a/2.0)*np.log(1.0+g))
    li = np.array(li)
    delta = np.log(grid[1]) - np.log(grid[0])
    lse = np.max(li + np.log(grid))
    integral = np.exp(lse) * np.sum(np.exp(li + np.log(grid) - lse)) * delta
    return float(np.log(integral))


# --- replace enumerate_models_BMA with this rank-safe version ---

def enumerate_models_BMA(X, y, feature_names, prior_inclusion=0.5, hyper_g_a=3.0, max_k=None):
    n, p = X.shape
    if max_k is None:
        max_k = p
    # Never try more predictors than n-1 (leaves DOF for noise)
    max_k = min(max_k, n-1, p)

    models = []; log_ml = []; priors = []; idxs = list(range(p))

    for k in range(1, max_k+1):
        for subset in itertools.combinations(idxs, k):
            Xs = X[:, subset]
            # add intercept
            Xsi = np.column_stack([np.ones(n), Xs])

            lm = log_marginal_hyper_g(Xsi, y, a=hyper_g_a)
            if not np.isfinite(lm) or lm < -1e11:
                # skip rank-deficient / degenerate
                continue

            log_ml.append(lm)
            m_prior = (prior_inclusion**k) * ((1 - prior_inclusion)**(p - k))
            priors.append(np.log(m_prior))
            models.append(subset)

    if len(models) == 0:
        # Fall back: empty result to avoid crash
        return {"models": [], "post_probs": np.array([]), "pip": {fn: 0.0 for fn in feature_names}}

    log_ml = np.array(log_ml); priors = np.array(priors)
    lu = log_ml + priors; m = np.max(lu)
    w = np.exp(lu - m); w = w / w.sum()

    pip = np.zeros(p)
    for weight, subset in zip(w, models):
        for j in subset:
            pip[j] += weight

    order = np.argsort(-w)
    return {
        "models": [models[i] for i in order],
        "post_probs": w[order],
        "pip": dict(zip(feature_names, pip))
    }

# ============== SHRINKAGE, ROBUST, OUTLIERS ==============

def _soft_thresh(z, t):
    if z > t: return z - t
    if z < -t: return z + t
    return 0.0

def lasso_cd(X, y, lam, max_iter=1000, tol=1e-6):
    # X should be standardized; y mean-centered (recommended)
    X = np.asarray(X); y = np.asarray(y)
    n, p = X.shape
    beta = np.zeros(p)
    col_norm2 = (X**2).sum(axis=0)
    for _ in range(max_iter):
        beta_old = beta.copy()
        r = y - X @ beta
        for j in range(p):
            rho = X[:, j].dot(r) + col_norm2[j]*beta[j]
            bj = _soft_thresh(rho/col_norm2[j], lam/col_norm2[j])
            r += X[:, j]*(beta[j] - bj)
            beta[j] = bj
        if np.linalg.norm(beta - beta_old) < tol*np.linalg.norm(beta_old + 1e-12):
            break
    return beta

def student_t_irls(X, y, nu=4.0, max_iter=100, tol=1e-6):
    X = np.asarray(X); y = np.asarray(y)
    n, p = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(max_iter):
        r = y - X @ beta
        s2 = (r @ r)/max(n-p,1)
        w = (nu+1) / (nu + (r**2)/max(s2,1e-12))
        W = np.diag(w)
        beta_new = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
        if np.linalg.norm(beta_new - beta) < tol*np.linalg.norm(beta + 1e-12):
            beta = beta_new
            break
        beta = beta_new
    r = y - X @ beta
    s2 = (r @ r)/max(n-p,1)
    w = (nu+1)/(nu + (r**2)/max(s2,1e-12))
    return beta, s2, w

def mean_shift_bma(X, y, top_k=6, prior_pi=0.2):
    # Outlier model averaging over top-K externally-studentized residuals
    n, p = X.shape
    XtX = X.T @ X; XtX_inv = np.linalg.inv(XtX); beta_hat = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ beta_hat
    h = np.sum(X * (X @ XtX_inv), axis=1)
    s2 = (e @ e)/(n - p)
    t_ext = e / np.sqrt(s2*(1 - h))
    idx = np.argsort(-np.abs(t_ext))[:min(top_k, n)]
    models = []; weights = []; incl = np.zeros(len(idx))
    for mask in range(1<<len(idx)):
        Z_cols = []
        for j,irow in enumerate(idx):
            if (mask >> j) & 1:
                col = np.zeros(n); col[irow] = 1.0
                Z_cols.append(col)
        if Z_cols:
            Z = np.column_stack(Z_cols); M = np.column_stack([X, Z])
        else:
            M = X
        beta_m = np.linalg.lstsq(M, y, rcond=None)[0]
        rss = float(np.sum((y - M @ beta_m)**2))
        k = M.shape[1]
        bic = n*np.log(rss/n) + k*np.log(n)
        prior_m = (prior_pi**len(Z_cols)) * ((1-prior_pi)**(len(idx)-len(Z_cols)))
        w = np.exp(-0.5*bic) * prior_m
        models.append([j for j in range(len(idx)) if (mask>>j)&1]); weights.append(w)
    weights = np.asarray(weights); weights = weights/weights.sum()
    for m, w in zip(models, weights):
        for j in m: incl[j] += w
    pip = {int(idx[j]): float(incl[j]) for j in range(len(idx))}
    return {"candidates": idx.tolist(), "post_weights": weights.tolist(), "pip": pip, "t_ext": t_ext.tolist()}

# ============== MIXED EFFECTS (random intercept) ==============

def random_intercept_by(y, X, groups, max_iter=200, tol=1e-6):
    # Simple REML-ish updates for a random-intercept model
    y = np.asarray(y).reshape(-1,1); X = np.asarray(X)
    import pandas as pd
    n, p = X.shape
    groups = pd.Series(groups).astype("category")
    G = int(groups.cat.categories.size)
    Z = np.zeros((n, G)); Z[np.arange(n), groups.cat.codes.values] = 1.0
    tau2 = np.var(y, ddof=1) * 0.1; sigma2 = np.var(y, ddof=1) * 0.9
    Xt = X.T; Zt = Z.T
    for _ in range(max_iter):
        s2 = sigma2
        A = np.eye(G) + (tau2/s2)*(Zt @ Z)
        A_inv = np.linalg.inv(A)
        V_inv = (1.0/s2)*np.eye(n) - (1.0/s2)*Z @ A_inv @ Zt * (1.0/s2)
        XtVinvX = Xt @ V_inv @ X
        beta = np.linalg.solve(XtVinvX, Xt @ V_inv @ y)
        resid = y - X @ beta
        b_hat = tau2 * (A_inv @ Zt @ (resid/s2))
        Eb2 = np.sum(b_hat**2) + tau2*np.trace(A_inv)
        Ee2 = float(resid.T @ resid) + sigma2*(n - p)
        tau2_new = Eb2 / G; sigma2_new = Ee2 / n
        if abs(tau2_new - tau2)/max(tau2,1e-8) < tol and abs(sigma2_new - sigma2)/max(sigma2,1e-8) < tol:
            tau2, sigma2 = tau2_new, sigma2_new; break
        tau2, sigma2 = max(tau2_new, 1e-12), max(sigma2_new, 1e-12)
    s2 = sigma2
    A = np.eye(G) + (tau2/s2)*(Zt @ Z); A_inv = np.linalg.inv(A)
    V_inv = (1.0/s2)*np.eye(n) - (1.0/s2)*Z @ A_inv @ Zt * (1.0/s2)
    XtVinvX = Xt @ V_inv @ X
    beta = np.linalg.solve(XtVinvX, Xt @ V_inv @ y)
    fitted = (X @ beta + Z @ b_hat).flatten()
    resid = (y.flatten() - fitted)
    return beta.flatten(), float(tau2), float(sigma2), V_inv, fitted, resid

# ============== DIAGNOSTICS & BASIC INFERENCE ==============

def ols_diagnostics(X, y, beta=None):
    X = np.asarray(X); y = np.asarray(y)
    n, p = X.shape
    if beta is None:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    h = np.sum(X * (X @ XtX_inv), axis=1)
    s2 = (e @ e)/(n - p)
    r_std = e / np.sqrt(s2*(1 - h))
    t_ext = r_std * np.sqrt((n - p - 1)/(n - p - r_std**2))
    cooks = (r_std**2) * h / (p * (1 - h))
    return {"beta": beta, "resid": e, "sigma2": s2, "leverage": h, "r_std": r_std, "t_ext": t_ext, "cooks": cooks}

def f_test(R, r, beta_hat, XtX_inv, s2, df_resid):
    diff = R @ beta_hat - r
    q = R.shape[0]
    F = float((diff.T @ np.linalg.inv(R @ XtX_inv @ R.T) @ diff) / (q * s2))
    # p-value needs scipy; return None if unavailable
    try:
        from scipy import stats
        p = 1 - stats.f.cdf(F, q, df_resid)
    except Exception:
        p = None
    return F, (float(p) if p is not None else None)
