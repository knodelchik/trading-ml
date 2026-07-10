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
- Same-window equal-weight buy-and-hold benchmark: **-30.0%**, maxDD -70.5%.
- Instructive negative result recorded in the ledger: the same signal traded
  every bar at threshold 0 lost -99.95% to costs. Turnover discipline
  matters more than model choice at this horizon.

Interpretation: directional timing added value over market beta net of
costs, but the evidence is one strategy variant over one 4.4-year window
with borderline significance — NOT a green light for real money. Next:
M2 (ensembles, better features, funding data), then Bybit demo paper
trading via the nautilus live node.
