"""Integer, per-asset transfer ledger with explicit open external accounts."""

from collections import defaultdict

from capclaims.domain.types import JournalEntry


class Ledger:
    def __init__(self) -> None:
        self._balances: dict[tuple[str, str], int] = defaultdict(int)
        self._entries: list[JournalEntry] = []

    @property
    def entries(self) -> tuple[JournalEntry, ...]:
        return tuple(self._entries)

    def balance(self, asset: str, account: str) -> int:
        return self._balances[(asset, account)]

    def post(self, entry: JournalEntry) -> None:
        if not entry.asset or not entry.source or not entry.destination:
            raise ValueError("asset and account names must be nonempty")
        if entry.source == entry.destination:
            raise ValueError("source and destination must differ")
        if isinstance(entry.amount, bool) or not isinstance(entry.amount, int) or entry.amount < 0:
            raise ValueError("transfer amount must be a nonnegative integer")
        if entry.amount == 0:
            return
        self._balances[(entry.asset, entry.source)] -= entry.amount
        self._balances[(entry.asset, entry.destination)] += entry.amount
        self._entries.append(entry)

    def validate_asset_conservation(self) -> bool:
        assets = {asset for asset, _ in self._balances}
        return all(sum(value for (name, _), value in self._balances.items() if name == asset) == 0 for asset in assets)
