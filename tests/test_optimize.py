import numpy as np
import pandas as pd
import pytest

from app.services.optimize import (
    OptimizeConfig,
    _select_top_assets,
    compute_ledoit_wolf_cov,
    optimize_weights,
)


def _synthetic_data(n_assets: int = 20, n_days: int = 120):
    rng = np.random.default_rng(42)
    symbols = [f"S{i:02d}" for i in range(n_assets)]
    # Create factor-like covariance
    factors = rng.normal(size=(n_days, 3))
    loadings = rng.normal(scale=0.5, size=(n_assets, 3))
    noise = rng.normal(scale=0.02, size=(n_days, n_assets))
    rets = factors @ loadings.T + noise
    returns = pd.DataFrame(rets, columns=symbols)
    mu = pd.Series(returns.mean().values, index=symbols)
    sectors = {s: ("A" if i % 2 == 0 else "B") for i, s in enumerate(symbols)}
    return symbols, returns, mu, sectors


def test_ledoit_wolf_cov_shape():
    symbols, returns, mu, sectors = _synthetic_data(15, 60)
    cov = compute_ledoit_wolf_cov(returns)
    assert cov.shape == (len(symbols), len(symbols))
    # PSD check (numerical): all eigenvalues non-negative within tolerance
    eig = np.linalg.eigvalsh(cov)
    assert eig.min() > -1e-8


@pytest.mark.parametrize("use_sectors", [True, False])
def test_optimize_basic_constraints(use_sectors: bool):
    symbols, returns, mu, sectors = _synthetic_data(20, 120)
    cov = compute_ledoit_wolf_cov(returns)
    prev = pd.Series(0.0, index=symbols)
    cfg = OptimizeConfig(risk_aversion=5.0, upper_bound=0.2, turnover_limit=0.5)
    res = optimize_weights(
        expected_returns=mu,
        cov_matrix=cov,
        prev_weights=prev,
        symbol_to_sector=sectors if use_sectors else None,
        config=cfg,
    )
    w = res["weights"]["weight"].values
    # Budget
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    # Bounds
    assert (w >= cfg.lower_bound - 1e-12).all() and (w <= cfg.upper_bound + 1e-12).all()
    # Cardinality cap
    assert (res["weights"]["weight"] > 1e-12).sum() <= cfg.max_assets
    # Sector caps (only if sectors used and cvxpy available); else skip soft check
    if use_sectors:
        A = pd.Series({k: 0.0 for k in set(sectors.values())})
        for sym, weight in res["weights"].itertuples(index=False):
            A[sectors[sym]] += weight
        assert (A <= cfg.sector_cap + 1e-6).all()


def test_turnover_limit_enforced():
    symbols, returns, mu, sectors = _synthetic_data(20, 120)
    cov = compute_ledoit_wolf_cov(returns)
    # Start from concentrated prev weights
    prev = pd.Series(0.0, index=symbols)
    prev.iloc[:5] = 0.2
    cfg = OptimizeConfig(risk_aversion=5.0, turnover_limit=0.10, upper_bound=0.25)
    res = optimize_weights(
        mu, cov, prev_weights=prev, symbol_to_sector=sectors, config=cfg
    )
    w = res["weights"].set_index("symbol")["weight"].reindex(symbols).fillna(0.0).values
    turnover = 0.5 * np.abs(w - prev.values).sum()
    # Allow small numerical slack
    assert turnover <= cfg.turnover_limit + 1e-3


def test_select_top_assets_basic():
    """Test dynamic selection of top-N assets by expected return."""
    mu = pd.Series([0.05, 0.10, 0.03, 0.08, 0.02], index=["A", "B", "C", "D", "E"])
    selected = _select_top_assets(mu, max_assets=3)
    assert len(selected) == 3
    assert "B" in selected.index  # Highest (0.10)
    assert "D" in selected.index  # Second (0.08)
    assert "A" in selected.index  # Third (0.05)
    assert selected.iloc[0] == 0.10  # Sorted descending


def test_select_top_assets_with_threshold():
    """Test selection with min_expected_return threshold."""
    mu = pd.Series([0.05, 0.10, 0.03, 0.08, 0.02], index=["A", "B", "C", "D", "E"])
    selected = _select_top_assets(mu, max_assets=10, min_expected_return=0.06)
    assert len(selected) == 2
    assert "B" in selected.index
    assert "D" in selected.index


def test_select_top_assets_threshold_too_strict():
    """Test that if threshold filters everything, we take at least top-1."""
    mu = pd.Series([0.05, 0.10, 0.03], index=["A", "B", "C"])
    selected = _select_top_assets(mu, max_assets=10, min_expected_return=0.50)
    assert len(selected) == 1
    assert "B" in selected.index  # Top asset


def test_optimize_weights_dynamic_selection():
    """Test that optimize_weights selects top-N and limits to max_assets."""
    # Create 100 assets with varying expected returns
    rng = np.random.default_rng(42)
    symbols = [f"S{i:03d}" for i in range(100)]
    mu = pd.Series(rng.uniform(-0.1, 0.1, 100), index=symbols)
    # Create covariance for all 100
    returns = pd.DataFrame(rng.normal(size=(60, 100)), columns=symbols)
    cov = compute_ledoit_wolf_cov(returns)

    cfg = OptimizeConfig(max_assets=10, risk_aversion=5.0)
    res = optimize_weights(mu, cov, config=cfg)

    # Should only have <= 10 assets in output
    assert len(res["weights"]) <= 10
    assert res["meta"]["n_selected"] <= 10
    # Weights should sum to 1.0
    assert np.isclose(res["weights"]["weight"].sum(), 1.0, atol=1e-6)
    # Selected assets should be top by expected return
    selected_symbols = set(res["weights"]["symbol"])
    top_10_by_return = set(mu.nlargest(10).index)
    # Should overlap significantly (allowing for optimization effects)
    assert len(selected_symbols & top_10_by_return) >= 5


def test_optimize_weights_less_than_max():
    """Test that N < max_assets is allowed when fewer candidates exist."""
    symbols = [f"S{i:02d}" for i in range(5)]
    mu = pd.Series([0.05, 0.10, 0.03, 0.08, 0.02], index=symbols)
    returns = pd.DataFrame(np.random.normal(size=(60, 5)), columns=symbols)
    cov = compute_ledoit_wolf_cov(returns)

    cfg = OptimizeConfig(max_assets=50, risk_aversion=5.0)
    res = optimize_weights(mu, cov, config=cfg)

    # Should have <= 5 assets (all available)
    assert len(res["weights"]) <= 5
    assert res["meta"]["n_selected"] <= 5
