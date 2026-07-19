"""Basis features: PIT alignment, warmup, the staleness cutoff, flag-off
identity, and ingest plumbing (URL formats, CSV/REST parsing) — no network."""

from __future__ import annotations

import hashlib
import io
import statistics
import urllib.error
import zipfile
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tests.conftest import make_config, make_raw
from tml.data import archives, basis, store
from tml.data.basis import PerpUnlisted, basis_path
from tml.features.basis import BASIS_COLS, EWM_SPAN, MAX_STALE_BARS, Z_WINDOW, add_basis_features
from tml.features.engine import build_features
from tml.live import refresh, state

T0 = datetime(2021, 1, 1, 0, 0, tzinfo=UTC)


def _write_basis(cfg, pair: str, rows: list[tuple[datetime, float]]) -> None:
    df = pl.DataFrame(
        {"ts": [r[0] for r in rows], "basis": [r[1] for r in rows], "pair": pair.replace("/", "")},
        schema_overrides={"ts": pl.Datetime("us", "UTC")},
    ).sort("ts")
    p = basis_path(cfg, pair)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p)


def _bars(pair: str, n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [T0 + timedelta(hours=i + 1) for i in range(n)],  # closes 01:00, 02:00, ...
            "pair": pair.replace("/", ""),
            "close": [100.0] * n,
        },
        schema_overrides={"ts": pl.Datetime("us", "UTC")},
    )


def _vals(n: int) -> list[float]:
    return [((i * 37) % 101 - 50) / 1e5 for i in range(n)]


# ---------------------------------------------------------------- features


def test_asof_uses_latest_kline_at_or_before_bar_close(tmp_path):
    cfg = make_config(tmp_path, pairs=("LTC/USDT",), basis=True)
    # a kline whose close carries millisecond jitter PAST the 08:00 bar close
    # must only become visible from the 09:00 bar onward
    _write_basis(
        cfg,
        "LTC/USDT",
        [
            (T0 + timedelta(hours=7), 0.001),
            (T0 + timedelta(hours=8, milliseconds=2), 0.002),
            (T0 + timedelta(hours=9), 0.003),
        ],
    )
    out = add_basis_features(_bars("LTC/USDT", 10), cfg)

    def at(hour: int, col: str = "basis_close") -> float:
        return out.filter(pl.col("ts") == T0 + timedelta(hours=hour))[col][0]

    assert at(8) == pytest.approx(0.001)
    assert at(9) == pytest.approx(0.003)  # 09:00 kline (exact) wins over 08:00.002
    assert at(7, "basis_known") == 1.0


def test_warmup_ewm_mom_z_and_known(tmp_path):
    cfg = make_config(tmp_path, pairs=("LTC/USDT",), basis=True)
    n = 800
    vals = _vals(n)
    _write_basis(cfg, "LTC/USDT", [(T0 + timedelta(hours=i + 1), v) for i, v in enumerate(vals)])
    out = add_basis_features(_bars("LTC/USDT", n), cfg).sort("ts")

    assert (out["basis_known"] == 1.0).all()  # fresh kline at every bar close
    assert out["basis_close"].to_list() == pytest.approx(vals, rel=1e-4)

    # ewm: manual adjust=False recursion, alpha = 2/(span+1)
    alpha = 2 / (EWM_SPAN + 1)
    e = vals[0]
    for v in vals[1:100]:
        e = alpha * v + (1 - alpha) * e
    assert out["basis_ewm"][99] == pytest.approx(e, rel=1e-3)

    # mom: 0-filled during the 24-bar warmup (known stays 1), then the diff
    assert out["basis_mom_24h"][10] == 0.0
    assert out["basis_mom_24h"][100] == pytest.approx(vals[100] - vals[76], rel=1e-3)

    # z: 0-filled until the 30d window fills (index Z_WINDOW-1), then real
    assert (out["basis_z"].slice(0, Z_WINDOW - 1) == 0.0).all()
    i = 780
    window = vals[i - Z_WINDOW + 1 : i + 1]
    expected = (vals[i] - statistics.fmean(window)) / (statistics.stdev(window) + 1e-12)
    assert out["basis_z"][i] == pytest.approx(expected, rel=1e-3)
    assert out["basis_z"].slice(Z_WINDOW - 1, None).abs().max() > 0.0


def test_prelisting_and_stale_series_are_neutral_and_flagged(tmp_path):
    """Pre-listing bars AND bars after the series stops (delisted perp or a
    failed live refresh) must both read basis_known=0 with zeroed values —
    a frozen last basis must never leak forward silently."""
    cfg = make_config(tmp_path, pairs=("LTC/USDT",), basis=True)
    _write_basis(cfg, "LTC/USDT", [(T0 + timedelta(hours=h), 0.001) for h in range(10, 51)])
    out = add_basis_features(_bars("LTC/USDT", 100), cfg)

    pre = out.filter(pl.col("ts") < T0 + timedelta(hours=10))
    live = out.filter(
        (pl.col("ts") >= T0 + timedelta(hours=10))
        & (pl.col("ts") <= T0 + timedelta(hours=50 + MAX_STALE_BARS))
    )
    dead = out.filter(pl.col("ts") > T0 + timedelta(hours=50 + MAX_STALE_BARS))
    assert dead.height > 0
    assert (pre["basis_known"] == 0.0).all()
    assert (live["basis_known"] == 1.0).all()
    assert (dead["basis_known"] == 0.0).all()
    for c in BASIS_COLS[:-1]:
        assert (pre[c] == 0.0).all()
        assert (dead[c] == 0.0).all()


def test_future_basis_perturbation_does_not_change_past(tmp_path):
    cfg = make_config(tmp_path, pairs=("LTC/USDT",), basis=True)
    rows = [(T0 + timedelta(hours=h + 1), v) for h, v in enumerate(_vals(96))]
    _write_basis(cfg, "LTC/USDT", rows)
    before = add_basis_features(_bars("LTC/USDT", 96), cfg)

    cut = T0 + timedelta(hours=48)
    perturbed = [(ts, v if ts <= cut else v * -7.0) for ts, v in rows]
    _write_basis(cfg, "LTC/USDT", perturbed)
    after = add_basis_features(_bars("LTC/USDT", 96), cfg)

    assert_frame_equal(before.filter(pl.col("ts") <= cut), after.filter(pl.col("ts") <= cut))


def _populate_raw(cfg, pairs: tuple[str, ...], n: int = 1200) -> None:
    for pair in pairs:
        df = make_raw(pair, n=n)
        for (year,), part in df.with_columns(year=pl.col("ts").dt.year()).group_by(
            "year", maintain_order=True
        ):
            store.write_raw_year(cfg, pair, int(year), part.drop("year"))


def _sha(df: pl.DataFrame) -> str:
    buf = io.BytesIO()
    df.write_parquet(buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def test_flag_off_is_byte_identical_and_flag_on_only_appends(tmp_path):
    pairs = ("AAA/USDT", "BBB/USDT")
    cfg_off = make_config(tmp_path, pairs=pairs)
    _populate_raw(cfg_off, pairs)
    off = build_features(cfg_off)
    assert not set(BASIS_COLS) & set(off.columns)
    assert _sha(off) == _sha(build_features(make_config(tmp_path, pairs=pairs, basis=False)))

    cfg_on = make_config(tmp_path, pairs=pairs, basis=True)
    _write_basis(
        cfg_on, "AAA/USDT", [(T0 + timedelta(hours=h + 1), v) for h, v in enumerate(_vals(1200))]
    )
    on = build_features(cfg_on)
    assert set(on.columns) - set(off.columns) == set(BASIS_COLS)
    assert_frame_equal(on.select(off.columns), off, check_exact=True)  # existing cols untouched
    assert (on.filter(pl.col("pair") == "BBBUSDT")["basis_known"] == 0.0).all()  # no perp
    assert (on.filter(pl.col("pair") == "AAAUSDT")["basis_known"] == 1.0).all()
    for c in BASIS_COLS:
        assert on.schema[c] == pl.Float32


# ------------------------------------------------------------------ ingest


def test_premium_archive_url_formatting():
    assert archives.premium_month_url("BTCUSDT", 2024, 1) == (
        "https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/"
        "BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"
    )
    assert archives.premium_month_url("ETHUSDT", 2025, 12).endswith("/ETHUSDT-1h-2025-12.zip")


def test_rest_url_formatting():
    assert basis.rest_url("BTCUSDT", 1704067200000) == (
        "https://fapi.binance.com/fapi/v1/premiumIndexKlines"
        "?symbol=BTCUSDT&interval=1h&startTime=1704067200000&limit=1500"
    )


HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
)


def _zip_bytes(csv_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("K.csv", csv_text)
    return buf.getvalue()


def test_fetch_premium_month_parses_verified_schema(monkeypatch):
    # first two rows of the real BTCUSDT-1h-2024-01.csv (verified download)
    csv_text = (
        HEADER + "\n"
        "1704067200000,0.00075030,0.00137721,0.00051408,0.00068158,0,1704070799999,0,720,0,0,0\n"
        "1704070800000,0.00062958,0.00164705,0.00050038,0.00083600,0,1704074399999,0,720,0,0,0\n"
    )
    monkeypatch.setattr(archives, "_get", lambda url: _zip_bytes(csv_text))
    df = archives.fetch_premium_month("BTCUSDT", 2024, 1)
    assert df is not None
    assert df["ts"].to_list() == [
        datetime(2024, 1, 1, 1, tzinfo=UTC),  # ts = bar CLOSE = open + 1h
        datetime(2024, 1, 1, 2, tzinfo=UTC),
    ]
    assert df["basis"].to_list() == pytest.approx([0.00068158, 0.00083600])
    assert df["pair"].to_list() == ["BTCUSDT"] * 2


def test_fetch_premium_month_handles_microsecond_epochs(monkeypatch):
    # 2025+ archives print epochs in MICROseconds — the digit-count branch
    csv_text = HEADER + "\n1735689600000000,1e-4,2e-4,0,1.5e-4,0,1735693199999999,0,720,0,0,0\n"
    monkeypatch.setattr(archives, "_get", lambda url: _zip_bytes(csv_text))
    df = archives.fetch_premium_month("BTCUSDT", 2025, 1)
    assert df is not None
    assert df["ts"].to_list() == [datetime(2025, 1, 1, 1, tzinfo=UTC)]


def _kline(open_ms: int, close_val: float, close_ms: int | None = None) -> list:
    if close_ms is None:
        close_ms = open_ms + 3_599_999
    return [open_ms, "0", "0", "0", f"{close_val:.8f}", "0", close_ms, "0", 720, "0", "0", "0"]


def test_fetch_premium_rest_paginates_and_drops_open_kline(monkeypatch):
    h = 3_600_000
    t0 = 1_704_067_200_000
    now_ms = t0 + 5 * h + 120_000  # two minutes past the 5th bar close
    monkeypatch.setattr(basis, "time", SimpleNamespace(time=lambda: now_ms / 1000))
    monkeypatch.setattr(basis, "REST_PAGE_LIMIT", 2)
    pages = {
        t0: [_kline(t0, 1e-4), _kline(t0 + h, 2e-4)],
        t0 + h + 1: [_kline(t0 + 2 * h, 3e-4), _kline(t0 + 3 * h, 4e-4)],
        # last page: one closed bar + the STILL-OPEN kline (close in the future)
        t0 + 3 * h + 1: [_kline(t0 + 4 * h, 5e-4), _kline(t0 + 5 * h, 6e-4)],
        t0 + 5 * h + 1: [],
    }
    monkeypatch.setattr(basis, "_rest_page", lambda sym, start: pages[start])

    df = basis.fetch_premium_rest("BTCUSDT", t0)
    assert df["ts"].to_list() == [
        datetime.fromtimestamp((t0 + (i + 1) * h) / 1000, tz=UTC) for i in range(5)
    ]
    assert df["basis"].to_list() == pytest.approx([1e-4, 2e-4, 3e-4, 4e-4, 5e-4])


def test_rest_unlisted_symbol_maps_to_perp_unlisted(monkeypatch):
    err = urllib.error.HTTPError(
        "u", 400, "Bad Request", None, io.BytesIO(b'{"code":-1121,"msg":"Invalid symbol."}')
    )

    def raise_err(url, timeout=0):
        raise err

    monkeypatch.setattr("urllib.request.urlopen", raise_err)
    with pytest.raises(PerpUnlisted):
        basis.fetch_premium_rest("NOPEUSDT", 1)


def test_run_basis_ingest_skips_unlisted(monkeypatch, tmp_path, capsys):
    cfg = make_config(tmp_path, pairs=("AAA/USDT", "BBB/USDT"), basis=True)

    def fake_fetch(cfg_, pair, months):
        if pair == "BBB/USDT":
            raise PerpUnlisted("BBBUSDT")
        return 7

    monkeypatch.setattr(basis, "fetch_basis_pair", fake_fetch)
    assert basis.run_basis_ingest(cfg) == 0
    out = capsys.readouterr().out
    assert "basis BBB/USDT: perp not listed — skipped" in out
    assert "basis AAA/USDT: +7 premium bars" in out


# ------------------------------------------------------------ live refresh


def test_run_basis_refresh_is_loud_never_silent(tmp_path, monkeypatch):
    """A failed pair is journaled (not raised), and the freshness audit
    journals any pair whose series still lags the staleness cutoff."""
    cfg = make_config(tmp_path, pairs=("AAA/USDT", "BBB/USDT"), basis=True)
    _write_basis(cfg, "AAA/USDT", [(datetime.now(UTC) - timedelta(days=14), 0.001)])

    def flaky(cfg_, pair):
        if pair == "BBB/USDT":
            raise RuntimeError("rest down")
        return 0

    monkeypatch.setattr(refresh, "refresh_basis_pair", flaky)
    assert refresh.run_basis_refresh(cfg) == 0  # the tick must go on

    errors = state.read_journal(cfg, "errors")
    by_stage = {e["stage"]: e for e in errors}
    assert by_stage["basis_refresh"]["pair"] == "BBB/USDT"
    assert "AAA/USDT" in by_stage["basis_stale"]["pairs"]
