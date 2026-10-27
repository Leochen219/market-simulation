"""Versioned configuration for the paper-aligned pilot."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

from capclaims.paper import PROFILE, SCHEMA_VERSION
from capclaims.paper.interface import validate_interface


@dataclass(frozen=True)
class PaperConfig:
    treatment: str = "GA"
    n_agents: int = 10
    wholesale_price: int = 10
    master_seed: int = 20261023
    seed_index: int = 0
    warmup_rounds: int = 5
    analyzed_rounds: int = 15
    history_window: int = 2
    maximum_orders: int = 5
    equality_crossing: str = "ge"
    self_match_policy: str = "allow"
    policy: str = "laya"
    decision_interface: str = "legacy_v1"
    instruction_profile: str = "explicit_v1"
    profile: str = PROFILE
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("n_agents", "wholesale_price", "master_seed", "seed_index", "warmup_rounds",
                     "analyzed_rounds", "history_window", "maximum_orders", "schema_version"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.profile != PROFILE or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported paper profile or schema version")
        if self.treatment not in {"NV", "G3", "GA"}:
            raise ValueError("treatment must be NV, G3, or GA")
        if isinstance(self.n_agents, bool) or not isinstance(self.n_agents, int) or self.n_agents < 1:
            raise ValueError("n_agents must be a positive integer")
        if self.treatment == "G3" and self.n_agents % 3:
            raise ValueError("G3 population must be divisible by three")
        if self.seed_index < 0:
            raise ValueError("seed_index must be nonnegative")
        if self.wholesale_price not in (10, 50):
            raise ValueError("wholesale_price must be 10 or 50")
        if self.warmup_rounds < 0 or self.analyzed_rounds < 1 or self.history_window < 0:
            raise ValueError("invalid round or history counts")
        if self.maximum_orders != 5:
            raise ValueError("paper profile allows at most five total orders")
        if self.equality_crossing not in {"ge", "gt"}:
            raise ValueError("equality_crossing must be ge or gt")
        if self.self_match_policy == "exclude":
            raise ValueError("self-match exclusion is not implemented; refusing unsupported sensitivity")
        if self.self_match_policy != "allow":
            raise ValueError("self_match_policy must be allow")
        if self.policy not in {"laya", "jev", "newsvendor", "random_legal", "noisy20"}:
            raise ValueError("paper policy must be laya, jev, newsvendor, random_legal, or noisy20")
        if self.instruction_profile not in {"explicit_v1", "matched_v2"}:
            raise ValueError("instruction_profile must be explicit_v1 or matched_v2")
        validate_interface(self.decision_interface)

    @property
    def total_rounds(self) -> int:
        return self.warmup_rounds + self.analyzed_rounds

    @classmethod
    def load(cls, path: str | Path) -> "PaperConfig":
        data: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("paper configuration must be a mapping")
        extra = set(data) - set(cls.__dataclass_fields__)
        if extra:
            raise ValueError(f"unknown paper configuration keys: {', '.join(sorted(extra))}")
        return cls(**data)
