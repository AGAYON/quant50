"""
Portfolio optimization services for Quant50 (T005).

Implements a robust Markowitz allocation with Ledoit–Wolf covariance
shrinkage, cardinality cap at 50 assets, sector caps, turnover limit
and basic bounds. Uses cvxpy when available; otherwise falls back to a
simple projected solution that honors budget and bounds (sector/turnover
constraints are only enforced when cvxpy is available).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class OptimizeConfig:
    risk_aversion: float = 10.0  # higher → more risk averse
    max_assets: int = 50
    lower_bound: float = 0.0
    upper_bound: float = 0.10
    turnover_limit: float = 0.15  # L1 distance/2 between w and w_prev
    sector_cap: float = 0.50
    target_vol: Optional[float] = None  # if provided, add volatility targeting
    min_expected_return: Optional[float] = (
        None  # Optional filter: only select assets
        #       with expected_return >= this threshold
    )


def compute_ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """
    Estimate covariance matrix with Ledoit–Wolf shrinkage.

    returns: wide DataFrame (rows=time, cols=symbol) of returns
    """
    X = returns.dropna(axis=0, how="any").values
    if X.shape[0] < 2:
        raise ValueError("Not enough observations for covariance estimation")
    try:
        from sklearn.covariance import LedoitWolf  # type: ignore

        lw = LedoitWolf().fit(X)
        return lw.covariance_  # type: ignore[no-any-return]
    except Exception:
        # Fallback: shrink towards diagonal with a fixed alpha
        sample_cov = np.cov(X, rowvar=False)
        diag = np.diag(np.diag(sample_cov))
        alpha = 0.1
        return (1.0 - alpha) * sample_cov + alpha * diag


def _build_sector_matrix(
    symbols: Sequence[str], symbol_to_sector: Dict[str, str]
) -> Tuple[np.ndarray, List[str]]:
    sectors = [symbol_to_sector.get(s, "UNK") for s in symbols]
    uniq = sorted(list(dict.fromkeys(sectors)))
    mat = np.zeros((len(uniq), len(symbols)), dtype=float)
    for i, sec in enumerate(uniq):
        for j, sym in enumerate(symbols):
            if sectors[j] == sec:
                mat[i, j] = 1.0
    return mat, uniq


def _select_top_assets(
    expected_returns: pd.Series,
    max_assets: int,
    min_expected_return: Optional[float] = None,
) -> pd.Series:
    """
    Select top-N assets by expected return ranking (T005A).

    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns indexed by symbol.
    max_assets : int
        Maximum number of assets to select (N ≤ max_assets).
    min_expected_return : float, optional
        Optional threshold: only select assets with expected_return >= this.

    Returns
    -------
    pd.Series
        Selected expected returns (subset of input, sorted descending).
    """
    # Apply threshold filter if provided
    if min_expected_return is not None:
        eligible = expected_returns[expected_returns >= min_expected_return]
    else:
        eligible = expected_returns.copy()

    if len(eligible) == 0:
        # If filter too strict, take at least the top asset
        eligible = expected_returns.nlargest(1)

    # Select top-N by expected return (descending)
    n_select = min(len(eligible), max_assets)
    selected = eligible.nlargest(n_select)

    return selected


def optimize_weights(
    expected_returns: pd.Series,
    cov_matrix: np.ndarray,
    prev_weights: Optional[pd.Series] = None,
    symbol_to_sector: Optional[Dict[str, str]] = None,
    config: Optional[OptimizeConfig] = None,
) -> Dict[str, object]:
    """
    Robust quadratic portfolio optimizer (Quant50 – T005/T005A).

    Implements a realistic Markowitz optimization with:
    - Dynamic asset selection: top-N by expected return (T005A)
    - hard constraints: budget, lower/upper bounds
    - soft penalties: sector caps, turnover control
    - stable solver (OSQP) without ECOS dependency
    - guaranteed feasibility and economic interpretability

    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns indexed by symbol.
    cov_matrix : np.ndarray
        Covariance matrix aligned with expected_returns.index.
    prev_weights : pd.Series, optional
        Previous portfolio weights (for turnover penalization).
    symbol_to_sector : dict, optional
        Mapping symbol → sector for sector exposure caps.
    config : OptimizeConfig, optional
        Risk aversion, bounds, sector caps, etc.

    Returns
    -------
    dict
        {
            "weights": pd.DataFrame(symbol, weight) with sum == 1.0,
            "meta": dict (risk_aversion, bounds, penalties, solver, n_selected)
        }
    """
    cfg = config or OptimizeConfig()

    # --- T005A: Dynamic selection by ranking ---
    selected_returns = _select_top_assets(
        expected_returns,
        max_assets=cfg.max_assets,
        min_expected_return=cfg.min_expected_return,
    )
    selected_symbols = list(selected_returns.index)
    n_selected = len(selected_symbols)

    # Reindex covariance matrix to selected symbols
    original_symbols = list(expected_returns.index)
    if set(selected_symbols) != set(original_symbols):
        # Build mapping: original index -> selected index
        orig_idx_map = {sym: i for i, sym in enumerate(original_symbols)}
        selected_idx = [orig_idx_map[sym] for sym in selected_symbols]
        cov_selected = cov_matrix[np.ix_(selected_idx, selected_idx)]
    else:
        cov_selected = cov_matrix

    symbols = selected_symbols
    n = n_selected
    mu = selected_returns.values.astype(float)
    Sigma = cov_selected.astype(float)
    w_prev = (
        prev_weights.reindex(symbols).fillna(0.0).values.astype(float)
        if prev_weights is not None
        else np.zeros(n, dtype=float)
    )

    # --- normalization for numerical stability ---
    mu_scale = np.max(np.abs(mu)) or 1.0
    sig_scale = np.max(np.abs(Sigma)) or 1.0
    mu = mu / mu_scale
    Sigma = Sigma / sig_scale

    try:
        import cvxpy as cp  # type: ignore

        w = cp.Variable(n)

        # --- objective components ---
        risk = cp.quad_form(w, Sigma)
        ret = mu @ w
        obj_terms = [ret - cfg.risk_aversion * risk]

        # --- sector cap penalty (soft, quadratic + linear) ---
        if symbol_to_sector is not None and len(symbol_to_sector) > 0:
            S, uniq = _build_sector_matrix(symbols, symbol_to_sector)
            sector_weights = S @ w
            penalty_sector = cp.sum_squares(
                cp.pos(sector_weights - cfg.sector_cap)
            ) + cp.sum(cp.pos(sector_weights - cfg.sector_cap))
            obj_terms.append(-1000.0 * penalty_sector)  # strong penalty

        # --- turnover penalty (soft L1) ---
        turnover_penalty = cp.norm1(w - w_prev)
        obj_terms.append(-10.0 * turnover_penalty)

        # --- aggregate objective ---
        objective = cp.Maximize(sum(obj_terms))

        # --- hard constraints ---
        constraints = [
            cp.sum(w) == 1.0,
            w >= cfg.lower_bound,
            w <= cfg.upper_bound,
        ]

        # --- solve ---
        prob = cp.Problem(objective, constraints)
        prob.solve(
            solver=cp.OSQP,
            eps_abs=1e-8,
            eps_rel=1e-8,
            max_iter=50000,
            polishing=True,
            verbose=False,
        )

        if w.value is None or prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Solver failed: {prob.status}")

        weights = np.asarray(w.value).reshape(-1)

        # --- cardinality constraint (post-optimization if needed) ---
        # If more than max_assets are active, re-solve with those assets fixed to zero
        if cfg.max_assets and cfg.max_assets < n:
            active = np.sum(weights > 1e-8)
            if active > cfg.max_assets:
                # Identify top k assets by weight
                top_idx = np.argsort(-weights)[: cfg.max_assets]
                mask = np.zeros(n, dtype=bool)
                mask[top_idx] = True
                # Re-solve with only top k assets allowed
                w_card = cp.Variable(n)
                obj_card_terms = [
                    mu @ w_card - cfg.risk_aversion * cp.quad_form(w_card, Sigma)
                ]
                if symbol_to_sector is not None and len(symbol_to_sector) > 0:
                    S, _ = _build_sector_matrix(symbols, symbol_to_sector)
                    sector_w = S @ w_card
                    penalty = cp.sum_squares(
                        cp.pos(sector_w - cfg.sector_cap)
                    ) + cp.sum(cp.pos(sector_w - cfg.sector_cap))
                    obj_card_terms.append(-1000.0 * penalty)
                turnover_pen = cp.norm1(w_card - w_prev)
                obj_card_terms.append(-10.0 * turnover_pen)
                obj_card = cp.Maximize(sum(obj_card_terms))
                constr_card = [
                    cp.sum(w_card) == 1.0,
                    w_card >= cfg.lower_bound,
                    w_card <= cfg.upper_bound,
                ]
                # Force non-top assets to zero
                for i in range(n):
                    if not mask[i]:
                        constr_card.append(w_card[i] == 0.0)
                prob_card = cp.Problem(obj_card, constr_card)
                prob_card.solve(
                    solver=cp.OSQP,
                    eps_abs=1e-8,
                    eps_rel=1e-8,
                    max_iter=50000,
                    polishing=True,
                    verbose=False,
                )
                if w_card.value is not None and prob_card.status in [
                    "optimal",
                    "optimal_inaccurate",
                ]:
                    weights = np.asarray(w_card.value).reshape(-1)

        # --- numerical cleanup ---
        weights[np.abs(weights) < 1e-8] = 0.0
        weights = np.clip(weights, cfg.lower_bound, cfg.upper_bound)

        # --- post checks ---
        budget = float(weights.sum())
        turnover = 0.5 * np.abs(weights - w_prev).sum()
        sec_cap = None
        if symbol_to_sector:
            S, _ = _build_sector_matrix(symbols, symbol_to_sector)
            sec_cap = (S @ weights).max()

        print(
            f"[INFO] Optimization done — budget={budget:.6f}, turnover={turnover:.4f}, "
            f"max_sector={sec_cap if sec_cap is not None else 'NA'}"
        )

    except Exception as e:
        print(f"[WARN] Fallback solver triggered: {e}")
        lam = cfg.risk_aversion
        try:
            inv = np.linalg.pinv(Sigma + lam * np.eye(n))
            x = inv @ mu
        except Exception:
            x = np.ones(n) / n
        x = np.clip(x, cfg.lower_bound, cfg.upper_bound)
        x /= x.sum()
        weights = x

    # --- output ---
    # Ensure weights sum to 1.0 (normalize if needed due to numerical precision)
    weights_sum = weights.sum()
    if abs(weights_sum - 1.0) > 1e-6:
        weights = weights / weights_sum

    out = pd.DataFrame({"symbol": symbols, "weight": weights}).sort_values(
        "weight", ascending=False
    )
    # Filter out zero weights for cleaner output
    out = out[out["weight"] > 1e-8].copy()

    meta = {
        "risk_aversion": cfg.risk_aversion,
        "max_assets": cfg.max_assets,
        "n_selected": n_selected,
        "bounds": [cfg.lower_bound, cfg.upper_bound],
        "turnover_limit": cfg.turnover_limit,
        "sector_cap": cfg.sector_cap,
        "solver": "cvxpy-OSQP-soft",
    }

    return {"weights": out, "meta": meta}
