# trading-ml

Rigorous ML crypto trading research pipeline. Milestone 1: hourly Binance spot
OHLCV → causal features → volatility-scaled labels → **purged walk-forward CV**
(López de Prado) → LightGBM baseline → honest statistics (Newey-West IC t-stats,
PSR, Deflated Sharpe over the trials ledger) → nautilus_trader portfolio
backtest with fees and slippage.

## Commands

```bash
uv sync --group dev          # environment (Python 3.13)
uv run tml all               # ingest → quality → features → labels → cv → backtest → report
uv run pytest -q             # leakage/causality/determinism test suite
```

Single config: `config/research.toml`. All artifacts land in `artifacts/`
(gitignored); every report number is traceable to an artifact file.

## Non-negotiables

- Rows are keyed by **bar close time**; the still-forming candle is dropped at
  ingestion. Features may look back only (prefix-invariance is unit-tested).
- Training samples whose labels overlap a test window are **purged**, plus an
  embargo margin. The backtest consumes exactly the CV's OOS predictions, so
  its retraining schedule cannot silently diverge.
- Every evaluated configuration appends to `artifacts/trials.jsonl`; the
  Deflated Sharpe Ratio is parameterized by that count and refuses to run
  without it. Deleting the ledger to "reset" the count is self-deception.
- Data gaps are journaled, never interpolated. Survivorship caveat: pairs were
  chosen in 2026 among still-listed majors.

## Status — Milestone 1 result (2026-07)

OOS 2022-01 → 2026-06, 18 walk-forward folds, 388,800 predictions.

- Cross-sectional ranking skill: pooled IC +0.033, **NW t-stat 0.58 — not
  statistically distinguishable from zero.**
- Net backtest (nautilus, 13 bps round-leg costs, long/flat hysteresis
  enter > 0.5σ / exit < 0): **+51.9% total, CAGR 9.9%, Sharpe 0.75, maxDD
  -19.4%**, fees paid ≈ 46% of initial capital.
- Execution-latency sensitivity (`execution_delay_bars = 1`, decide at bar
  t, fill at t+1 close): **+39.4%, Sharpe 0.61** — degrades but survives.
- Same-window equal-weight buy-and-hold benchmark: **-30.0%**, maxDD -70.5%.
- **Net DSR: 0.959 over the two serious execution variants; 0.054 once the
  deliberately-naive threshold-0 trial joins the family** (the expected-max
  benchmark is extremely sensitive to degenerate trials — both numbers are
  reported rather than silently choosing one). Ledger counts distinct
  same-stage, same-estimator configurations; re-runs are not new trials.
- Instructive negative result recorded in the ledger: the same signal traded
  every bar at threshold 0 lost -99.95% to costs. Turnover discipline
  matters more than model choice at this horizon.

Interpretation: directional timing added value over market beta net of
costs and survives one bar of execution latency, but the evidence rests on
one signal family over one 4.4-year window and does not robustly clear the
deflated significance bar — promising, NOT proven, and NOT a green light
for real money.

## Milestone 2 (2026-07): funding features + tuning + ensemble

Added Binance perp funding-rate features (69.7k settlements, point-in-time
asof joins), multi-scale horizons to 336h, Optuna tuning confined to the
first train window with inner purged splits, and a 3-member ensemble.
Per-model OOS diagnostics (same 18 folds):

| model | pooled IC | NW t-stat |
|---|---|---|
| LightGBM (tuned, funding) | +0.038 | **1.92** |
| CatBoost | +0.039 | 1.35 |
| TabPFN (recent-10k window) | +0.015 | -0.29 |
| rank-average (diagnostic) | **+0.049** | 1.29 |
| mean ensemble (traded) | +0.032 | 0.10 |

Findings: funding + tuning raised single-model NW t from 0.58 (M1) to 1.92
— near the 1.96 significance bar. TabPFN's 10k-recency view is weak on this
noise level and DILUTES the mean ensemble (traded mean underperformed both
boosters: net +22.8%, Sharpe 0.32); rank-average is the strongest ranking
signal. Engineering notes: TabPFN runs in a subprocess (duplicate-OpenMP
deadlock with boosters on macOS); TabPFN weights are NON-COMMERCIAL
licensed — must be dropped before real-money use.
