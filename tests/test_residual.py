"""BTC-residual features: known-beta recovery, warmup flags, prefix
invariance THROUGH the cross-pair BTC path, cs z-score causality, and
flag-off output identical to the pre-flag pipeline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from tests.conftest import make_config, make_raw
from tml.data import store
from tml.features.cross_sectional import add_cs_zscores
from tml.features.engine import build_features, pair_features
from tml.features.residual import (
    BETA_WINDOW,
    CORR_WINDOW,
    CS_Z_COLS,
    MOM_WINDOWS,
    RESID_COLS,
    add_residual_features,
)

START = datetime(2021, 1, 1, 1, 0, tzinfo=UTC)  # first bar CLOSE time
FIRST_KNOWN = BETA_WINDOW + max(MOM_WINDOWS) - 1  # 720-bar beta + 168-bar resid sum


def _raw_from_returns(pair: str, r: np.ndarray) -> pl.DataFrame:
    """OHLCV frame whose 1-bar log returns are exactly r[1:]."""
    close = 100.0 * np.exp(np.cumsum(r))
    n = len(r)
    return pl.DataFrame(
        {
            "ts": [START + timedelta(hours=i) for i in range(n)],
            "open": np.concatenate([[100.0], close[:-1]]),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.ones(n),
            "pair": store.pair_key(pair),
        },
        schema_overrides={"ts": pl.Datetime("us", "UTC")},
    )


def _write_store(cfg, frames: dict[str, pl.DataFrame]) -> None:
    for pair, df in frames.items():
        for (year,), part in df.with_columns(year=pl.col("ts").dt.year()).group_by(
            "year", maintain_order=True
        ):
            store.write_raw_year(cfg, pair, int(year), part.drop("year"))


def _bars(raw: pl.DataFrame) -> pl.DataFrame:
    return raw.select("ts", "pair", "close")


def test_rolling_beta_recovers_known_beta(tmp_path):
    n = 1000
    cfg = make_config(tmp_path, pairs=("BTC/USDT", "AAA/USDT"), btc_residual=True)
    rng = np.random.default_rng(7)
    r_btc = rng.normal(0.0, 0.01, n)
    raw_btc = _raw_from_returns("BTC/USDT", r_btc)
    raw_aaa = _raw_from_returns("AAA/USDT", 2.0 * r_btc)  # exact beta 2, zero residual
    _write_store(cfg, {"BTC/USDT": raw_btc, "AAA/USDT": raw_aaa})

    out = add_residual_features(_bars(raw_aaa), cfg)
    known = out.filter(pl.col("resid_known") == 1.0)
    assert known.height == n - FIRST_KNOWN
    np.testing.assert_allclose(known["beta_btc"].to_numpy(), 2.0, atol=1e-3)
    np.testing.assert_allclose(known["corr_btc"].to_numpy(), 1.0, atol=1e-3)
    for h in MOM_WINDOWS:
        assert known[f"resid_mom_{h}"].abs().max() < 1e-3  # residual of an exact fit is ~0


def test_warmup_rows_are_neutral_and_flagged(tmp_path):
    n = 1000
    cfg = make_config(tmp_path, pairs=("BTC/USDT", "AAA/USDT"), btc_residual=True)
    rng = np.random.default_rng(11)
    r_btc = rng.normal(0.0, 0.01, n)
    raw_btc = _raw_from_returns("BTC/USDT", r_btc)
    raw_aaa = _raw_from_returns("AAA/USDT", 0.7 * r_btc + rng.normal(0.0, 0.005, n))
    _write_store(cfg, {"BTC/USDT": raw_btc, "AAA/USDT": raw_aaa})

    out = add_residual_features(_bars(raw_aaa), cfg)
    assert int(out.null_count().sum_horizontal().item()) == 0
    for c in RESID_COLS:
        assert out.schema[c] == pl.Float32, f"{c} is {out.schema[c]}"
    # flag turns on exactly when the slowest window (720h beta + 168h sum) forms
    assert out["resid_known"].to_list() == [0.0] * FIRST_KNOWN + [1.0] * (n - FIRST_KNOWN)
    # unformed windows are neutral zero (corr forms at 168h, before the beta;
    # formed-but-not-all-known values keep their real value, funding-style)
    pre_beta = out.slice(0, BETA_WINDOW)
    for c in ("beta_btc", "resid_mom_24", "resid_mom_72", "resid_mom_168"):
        assert (pre_beta[c] == 0.0).all(), f"{c} not neutral during beta warmup"
    assert (out.slice(0, CORR_WINDOW)["corr_btc"] == 0.0).all()
    assert (out.slice(CORR_WINDOW, 50)["corr_btc"] != 0.0).any()


def test_prefix_invariance_truncation_including_btc(tmp_path):
    n, cut = 1100, 950
    rng = np.random.default_rng(3)
    r_btc = rng.normal(0.0, 0.01, n)
    r_aaa = 1.3 * r_btc + rng.normal(0.0, 0.004, n)
    raw_btc = _raw_from_returns("BTC/USDT", r_btc)
    raw_aaa = _raw_from_returns("AAA/USDT", r_aaa)

    cfg_full = make_config(tmp_path / "full", pairs=("BTC/USDT", "AAA/USDT"), btc_residual=True)
    cfg_trunc = make_config(tmp_path / "trunc", pairs=("BTC/USDT", "AAA/USDT"), btc_residual=True)
    _write_store(cfg_full, {"BTC/USDT": raw_btc, "AAA/USDT": raw_aaa})
    _write_store(cfg_trunc, {"BTC/USDT": raw_btc.slice(0, cut), "AAA/USDT": raw_aaa.slice(0, cut)})

    full = add_residual_features(_bars(raw_aaa), cfg_full)
    trunc = add_residual_features(_bars(raw_aaa.slice(0, cut)), cfg_trunc)
    assert_frame_equal(full.slice(0, cut), trunc)


def test_btc_perturbation_after_cut_does_not_change_past(tmp_path):
    """The cross-pair leak check: scaling BTC AFTER the cut must not move
    any other pair's residual features AT or before it."""
    n, cut = 1100, 900
    rng = np.random.default_rng(5)
    r_btc = rng.normal(0.0, 0.01, n)
    raw_btc = _raw_from_returns("BTC/USDT", r_btc)
    raw_aaa = _raw_from_returns("AAA/USDT", 1.5 * r_btc + rng.normal(0.0, 0.004, n))

    btc_perturbed = raw_btc.with_columns(
        [
            pl.when(pl.arange(0, n) >= cut).then(pl.col(c) * 3.0).otherwise(pl.col(c)).alias(c)
            for c in ("open", "high", "low", "close")
        ]
    )
    cfg_a = make_config(tmp_path / "a", pairs=("BTC/USDT", "AAA/USDT"), btc_residual=True)
    cfg_b = make_config(tmp_path / "b", pairs=("BTC/USDT", "AAA/USDT"), btc_residual=True)
    _write_store(cfg_a, {"BTC/USDT": raw_btc, "AAA/USDT": raw_aaa})
    _write_store(cfg_b, {"BTC/USDT": btc_perturbed, "AAA/USDT": raw_aaa})

    before = add_residual_features(_bars(raw_aaa), cfg_a)
    after = add_residual_features(_bars(raw_aaa), cfg_b)
    cut_ts = START + timedelta(hours=cut)  # first perturbed bar's close
    assert_frame_equal(
        before.filter(pl.col("ts") < cut_ts), after.filter(pl.col("ts") < cut_ts)
    )
    # sanity: the perturbation IS visible afterwards (the test has teeth)
    assert not before.filter(pl.col("ts") >= cut_ts).equals(after.filter(pl.col("ts") >= cut_ts))


def test_cs_zscores_use_only_same_timestamp():
    fc = make_config("/tmp").features
    frames = [pair_features(make_raw(p, n=600), fc) for p in ("AAA/USDT", "BBB/USDT", "CCC/USDT")]
    df = pl.concat(frames).sort("ts", "pair")
    z = add_cs_zscores(df, ["mom_24"])

    ts_split = df["ts"].unique().sort()[300]
    perturbed = df.with_columns(
        pl.when(pl.col("ts") > ts_split).then(pl.col("mom_24") * -3.0).otherwise(pl.col("mom_24")).alias("mom_24")
    )
    z_p = add_cs_zscores(perturbed, ["mom_24"])
    assert_frame_equal(z.filter(pl.col("ts") <= ts_split), z_p.filter(pl.col("ts") <= ts_split))

    # per-ts standardization: mean ~0, sd ~1 wherever >= 3 pairs are present
    stats = (
        z.group_by("ts")
        .agg(m=pl.col("cs_z_mom_24").mean(), s=pl.col("cs_z_mom_24").std(), n=pl.len())
        .filter(pl.col("n") >= 3)
    )
    assert stats.height > 0
    assert stats["m"].abs().max() < 1e-4
    assert (stats["s"] - 1.0).abs().max() < 1e-3


def test_flag_off_output_is_unchanged(tmp_path):
    pairs = ("BTC/USDT", "AAA/USDT", "BBB/USDT")
    overrides = {"vol_spans": [12, 24, 168]}  # cs_z needs vol_ewm_168
    cfg_off = make_config(tmp_path, pairs=pairs, features_overrides=overrides)
    cfg_on = make_config(tmp_path, pairs=pairs, btc_residual=True, features_overrides=overrides)
    _write_store(cfg_off, {p: make_raw(p, n=2000) for p in pairs})  # same root/store for both

    f_off = build_features(cfg_off)
    f_on = build_features(cfg_on)

    new_cols = set(f_on.columns) - set(f_off.columns)
    assert new_cols == set(RESID_COLS) | {f"cs_z_{c}" for c in CS_Z_COLS}
    # flag off adds nothing and changes nothing: identical frame, column order included
    assert_frame_equal(f_off, f_on.select(f_off.columns))
    assert not any(c in f_off.columns for c in new_cols)

    # flag on keeps the engine's discipline: no nulls, float32 model inputs
    assert int(f_on.null_count().sum_horizontal().item()) == 0
    for c in f_on.columns:
        if c not in ("ts", "pair", "close"):
            assert f_on.schema[c] == pl.Float32, f"{c} is {f_on.schema[c]}"
