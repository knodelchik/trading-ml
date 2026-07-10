"""Configuration loading and validation.

The TOML file at config/research.toml is the single source of truth.
Cross-field invariants that protect leakage discipline (purge width vs
label horizon) are enforced here so a bad config fails before any stage runs.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

TIMEFRAME_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DataConfig(_Frozen):
    exchange: str
    timeframe: Literal["1m", "5m", "15m", "1h", "4h", "1d"]
    since: datetime
    pairs: tuple[str, ...]

    @field_validator("pairs")
    @classmethod
    def _pairs_non_empty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("data.pairs must be non-empty")
        if len(set(v)) != len(v):
            raise ValueError("data.pairs contains duplicates")
        return v

    @field_validator("since")
    @classmethod
    def _since_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("data.since must be timezone-aware (UTC)")
        return v.astimezone(UTC)

    @property
    def bar_ms(self) -> int:
        return TIMEFRAME_MS[self.timeframe]


class FeaturesConfig(_Frozen):
    ret_horizons: tuple[int, ...]
    mom_horizons: tuple[int, ...]
    vol_spans: tuple[int, ...]
    rsi_period: int
    atr_period: int
    donchian_window: int
    volume_z_window: int
    cs_rank_cols: tuple[str, ...]

    @property
    def max_lookback(self) -> int:
        """Warmup width: rows younger than this (per pair) are dropped."""
        return max(
            max(self.ret_horizons),
            max(self.mom_horizons),
            max(self.vol_spans),
            self.rsi_period,
            self.atr_period,
            self.donchian_window,
            self.volume_z_window,
        )


class LabelsConfig(_Frozen):
    kind: Literal["fixed_horizon_volscaled"]  # "triple_barrier" arrives in M2
    horizon_bars: int
    vol_span: int

    @field_validator("horizon_bars", "vol_span")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("labels horizons must be positive")
        return v


class CVConfig(_Frozen):
    train_min_bars: int
    test_span_bars: int
    step_bars: int
    purge_bars: int
    embargo_bars: int

    @field_validator("train_min_bars", "test_span_bars", "step_bars")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("cv window sizes must be positive")
        return v


class LGBMConfig(_Frozen):
    seed: int
    num_leaves: int
    learning_rate: float
    n_estimators: int
    min_data_in_leaf: int
    feature_fraction: float
    num_threads: int


class ModelConfig(_Frozen):
    lgbm: LGBMConfig


class BacktestConfig(_Frozen):
    enter_threshold: float
    exit_threshold: float
    taker_fee_bps: float
    slippage_bps: float
    execution_delay_bars: int
    initial_cash_usdt: float

    @model_validator(mode="after")
    def _hysteresis(self) -> BacktestConfig:
        if self.enter_threshold < self.exit_threshold:
            raise ValueError("enter_threshold must be >= exit_threshold (hysteresis band)")
        return self


class Config(_Frozen):
    data: DataConfig
    features: FeaturesConfig
    labels: LabelsConfig
    cv: CVConfig
    model: ModelConfig
    backtest: BacktestConfig
    root: Path

    @model_validator(mode="after")
    def _leakage_invariants(self) -> Config:
        if self.cv.purge_bars < self.labels.horizon_bars:
            raise ValueError(
                f"cv.purge_bars ({self.cv.purge_bars}) must be >= labels.horizon_bars "
                f"({self.labels.horizon_bars}): training samples whose labels overlap the "
                "test window must be purged"
            )
        return self

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    def config_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"root"})
        canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()[:16]


def load_config(path: str | Path = "config/research.toml") -> Config:
    p = Path(path).resolve()
    with open(p, "rb") as f:
        raw = tomllib.load(f)
    raw["root"] = p.parent.parent  # config/ lives directly under the project root
    return Config.model_validate(raw)
