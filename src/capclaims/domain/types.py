"""Domain records shared by the auction and settlement layers."""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Order:
    agent_id: str
    side: str
    qty: int
    limit_cents: int


@dataclass(frozen=True)
class Trade:
    buyer_id: str
    seller_id: str
    qty: int
    price_cents: int


@dataclass(frozen=True)
class AuctionResult:
    price_cents: int | None
    trades: tuple[Trade, ...]
    net_allocations: dict[str, int]
    tie_ranks: dict[str, int]


@dataclass(frozen=True)
class SettlementResult:
    profits_cents: dict[str, int]
    fulfilled: dict[str, int]
    leftovers: dict[str, int]
    shortages: dict[str, int]
    entries: tuple["JournalEntry", ...]
    post_sweep_balances: dict[str, int]
    issued_units: int
    redeemed_units: int
    burned_units: int


@dataclass(frozen=True)
class FrozenRoundState:
    valuations: Mapping[str, int]
    demands: Mapping[str, int]
    initial_quantities: Mapping[str, int]


@dataclass(frozen=True)
class JournalEntry:
    source: str
    destination: str
    asset: str
    amount: int
    reason: str


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    value: int | str
    label: str


@dataclass(frozen=True)
class TradeAction:
    side: str = "hold"
    qty: int = 0
    limit_cents: int | None = None
