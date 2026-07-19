"""Shared fixtures: deterministic synthetic OHLCV and a small fast Config."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from tml.config import Config
from tml.data import store

START = datetime(2021, 1, 1, 1, 0, tzinfo=UTC)  # first bar CLOSE time
# 8000 bars -> ~29k OOS rows in the null-label test -> detection floor
# 2/sqrt(N) ~= 0.012, well below the ~0.03 IC magnitude we defend on real
# data (2000 bars gave a 0.027 floor — too weak to catch signal-sized leaks)
N_BARS = 8000
PAIRS = ("AAA/USDT", "BBB/USDT", "CCC/USDT", "DDD/USDT")


def make_raw(pair: str, n: int = N_BARS, seed: int | None = None, start: datetime = START) -> pl.DataFrame:
    """Deterministic geometric random walk with plausible OHLCV shape."""
    if seed is None:
        # stable across processes (unlike built-in hash(), which is salted)
        seed = int.from_bytes(hashlib.sha256(pair.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(r))
    spread = np.abs(rng.normal(0, 0.004, n))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = np.exp(rng.normal(10, 1, n))
    ts = [start + timedelta(hours=i) for i in range(n)]
    return pl.DataFrame(
        {
            "ts": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "pair": store.pair_key(pair),
        },
        schema_overrides={"ts": pl.Datetime("us", "UTC")},
    )


def make_config(
    root,
    pairs: tuple[str, ...] = PAIRS,
    funding: bool = False,
    universe: dict | None = None,
    btc_residual: bool = False,
    features_overrides: dict | None = None,
) -> Config:
    return Config.model_validate(
        {
            **({"universe": universe} if universe else {}),
            "data": {
                "exchange": "binance",
                "timeframe": "1h",
                "since": "2021-01-01T00:00:00Z",
                "pairs": list(pairs),
                "funding": funding,
                "btc_residual": btc_residual,
            },
            "features": {
                "ret_horizons": [1, 3, 6],
                "mom_horizons": [24, 72],
                "vol_spans": [12, 24],
                "rsi_period": 14,
                "atr_period": 14,
                "donchian_window": 48,
                "volume_z_window": 48,
                "cs_rank_cols": ["mom_24", "mom_72", "vol_ewm_24"],
                **(features_overrides or {}),
            },
            "labels": {"kind": "fixed_horizon_volscaled", "horizon_bars": 6, "vol_span": 24},
            "cv": {
                "train_min_bars": 600,
                "test_span_bars": 200,
                "step_bars": 200,
                "purge_bars": 6,
                "embargo_bars": 6,
            },
            "model": {
                "lgbm": {
                    "seed": 42,
                    "num_leaves": 15,
                    "learning_rate": 0.1,
                    "n_estimators": 50,
                    "min_data_in_leaf": 20,
                    "feature_fraction": 0.8,
                    "num_threads": 2,
                }
            },
            "backtest": {
                "enter_threshold": 0.0,
                "exit_threshold": 0.0,
                "execution_delay_bars": 0,
                "initial_cash_usdt": 100000.0,
            },
            "execution": {
                "mode": "taker",
                "maker_fee_bps": 2.0,
                "taker_fee_bps": 10.0,
                "slippage_bps": 3.0,
                "limit_ttl_bars": 2,
            },
            "root": str(root),
        }
    )


@pytest.fixture
def synth_cfg(tmp_path) -> Config:
    return make_config(tmp_path)


@pytest.fixture
def synth_store(synth_cfg) -> Config:
    """Config whose artifact store is populated with synthetic raw bars."""
    for pair in synth_cfg.data.pairs:
        df = make_raw(pair)
        for (year,), part in df.with_columns(year=pl.col("ts").dt.year()).group_by(
            "year", maintain_order=True
        ):
            store.write_raw_year(synth_cfg, pair, int(year), part.drop("year"))
    return synth_cfg
