"""Validated configuration for one market session."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunConfig:
    master_seed: int = 20261023
    seed_index: int = 0
    n_agents: int = 10
    rounds: int = 15
    wholesale_price: int = 10
    market_open: bool = True
    policy: str = "grid_rule"
    mechanism_version: str = "single_order_v1"
    initial_cash_cents: int = 10_000_000

    def __post_init__(self) -> None:
        int_fields = ("master_seed", "seed_index", "n_agents", "rounds", "wholesale_price", "initial_cash_cents")
        for name in int_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.n_agents != 10:
            raise ValueError("n_agents must be 10 in v1")
        if self.rounds < 1:
            raise ValueError("rounds must be at least 1")
        if self.wholesale_price not in (10, 50):
            raise ValueError("wholesale_price must be 10 or 50")
        if self.seed_index < 0:
            raise ValueError("seed_index must be nonnegative")
        if self.initial_cash_cents < 400 * self.wholesale_price * 100 + 400 * 10_000:
            raise ValueError("initial_cash_cents is below the maximum action bound")
        if not isinstance(self.market_open, bool):
            raise ValueError("market_open must be a boolean")
        if self.policy not in {"random", "grid_rule", "noisy_rule", "laya", "jev"}:
            raise ValueError("policy must be random, grid_rule, noisy_rule, laya, or jev")
        if self.mechanism_version != "single_order_v1":
            raise ValueError("mechanism_version must be single_order_v1")

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        try:
            data: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML configuration: {exc}") from exc
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("configuration YAML must contain a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown configuration keys: {', '.join(sorted(map(str, unknown)))}")
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(f"invalid configuration fields: {exc}") from exc
