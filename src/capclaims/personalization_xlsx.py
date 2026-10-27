"""Read-only adapter for the paper's exact Data_vf.xlsx workbook structure."""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from capclaims.personalization import Episode, SourceManifest

_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
_SHEETS = ("NV 50", "G3 50", "GA 50", "NV 10", "G3 10", "GA 10")


def _col_index(ref: str) -> int:
    name = re.match(r"[A-Z]+", ref)
    if not name:
        raise ValueError("invalid XLSX cell reference")
    value = 0
    for char in name.group(0):
        value = value * 26 + ord(char) - 64
    return value - 1


def _inner_xlsx_bytes(path: str | Path) -> bytes:
    raw = Path(path).read_bytes()
    source = io.BytesIO(raw)
    if not zipfile.is_zipfile(source):
        raise ValueError("source must be an XLSX workbook or ZIP containing Data_vf.xlsx")
    source.seek(0)
    with zipfile.ZipFile(source) as archive:
        members = set(archive.namelist())
        # An XLSX file is itself a ZIP. Distinguish it from the supplied outer archive.
        if "xl/workbook.xml" in members:
            return raw
        candidates = [name for name in members if name.endswith("Data_vf.xlsx")]
        if len(candidates) != 1:
            raise ValueError("expected exactly one Data_vf.xlsx workbook member")
        return archive.read(candidates[0])


def _workbook(path: str | Path):
    source = io.BytesIO(_inner_xlsx_bytes(path))
    archive = zipfile.ZipFile(source)
    bad = [name for name in archive.namelist() if name.lower().endswith((".vba", ".exe", ".py"))
           or (name.lower().endswith(".bin") and "printersettings" not in name.lower())]
    if bad:
        archive.close()
        raise ValueError("workbook contains unsupported embedded executable content")
    try:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        shared_root = None
    shared = [] if shared_root is None else ["".join(t.text or "" for t in si.iter(f"{{{_MAIN}}}t"))
                                               for si in shared_root.findall(f"{{{_MAIN}}}si")]
    book = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map = {r.attrib["Id"]: r.attrib["Target"] for r in rels.findall(f"{{{_PKG}}}Relationship")}
    names = {}
    for sheet in book.find(f"{{{_MAIN}}}sheets"):
        rid = sheet.attrib[f"{{{_REL}}}id"]
        target = rel_map[rid].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        names[sheet.attrib["name"]] = target
    return archive, shared, names


def _sheet_rows(archive: zipfile.ZipFile, member: str, shared: list[str]) -> list[list[Any]]:
    root = ET.fromstring(archive.read(member))
    data = root.find(f"{{{_MAIN}}}sheetData")
    rows: list[list[Any]] = []
    for row in data or ():
        cells: dict[int, Any] = {}
        for cell in row.findall(f"{{{_MAIN}}}c"):
            col = _col_index(cell.attrib.get("r", ""))
            value = cell.find(f"{{{_MAIN}}}v")
            raw = "" if value is None or value.text is None else value.text
            if cell.attrib.get("t") == "s":
                raw = shared[int(raw)]
            elif cell.attrib.get("t") == "inlineStr":
                raw = "".join(t.text or "" for t in cell.iter(f"{{{_MAIN}}}t"))
            cells[col] = raw
        if cells:
            row_values = [""] * (max(cells) + 1)
            for col, value in cells.items():
                row_values[col] = value
            rows.append(row_values)
    return rows


def _number(value: Any, field: str, *, required: bool = True) -> int | None:
    if value in (None, "") and not required:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric workbook field {field}") from exc
    if isinstance(value, float) and value != number:
        raise ValueError(f"noninteger workbook field {field}")
    return number


def inspect_workbook(path: str | Path) -> dict[str, Any]:
    """Return structural metadata only: sheet names, headers, and data row counts."""
    archive, shared, sheet_members = _workbook(path)
    try:
        result = []
        for name in _SHEETS:
            if name not in sheet_members:
                raise ValueError(f"expected worksheet missing: {name}")
            rows = _sheet_rows(archive, sheet_members[name], shared)
            if not rows:
                raise ValueError(f"worksheet is empty: {name}")
            headers = [str(item) for item in rows[0]]
            result.append({"sheet": name, "header_count": len(headers), "headers": headers,
                "data_rows": max(0, len(rows) - 1)})
        return {"workbook": Path(path).name, "sheets": result}
    finally:
        archive.close()


def load_paper_episodes(path: str | Path, *, intended_use_authorized: bool = False,
                        provider_transmission_permitted: bool = False,
                        codebook_verified: bool = False) -> tuple[SourceManifest, list[Episode]]:
    """Read validated prior behavior; manifest attestations must be supplied separately."""
    path = Path(path)
    checksum = hashlib.sha256(_inner_xlsx_bytes(path)).hexdigest()
    archive, shared, sheet_members = _workbook(path)
    episodes: list[Episode] = []
    seen: set[tuple[str, int]] = set()
    try:
        for sheet in _SHEETS:
            if sheet not in sheet_members:
                raise ValueError(f"expected worksheet missing: {sheet}")
            rows = _sheet_rows(archive, sheet_members[sheet], shared)
            if not rows:
                raise ValueError(f"worksheet is empty: {sheet}")
            headers = [str(value) for value in rows[0]]
            expected = (["Player", "Round", "Key1", "Order", "Sales Price", "Demand", "Profit ($)",
                         "Excess Initial (#)", "Shortage Initial (#)"] if sheet.startswith("NV") else
                        ["Player", "Round", "PR Key", "Room", "Sales Price", "Purchase Cost", "Order",
                         "Actual Demand", "Sales (#)", "Profit Projection ()", "Sales New (#)",
                         "Excess Initial (#)", "Shortage Initial (#)", "Trade Cost", "Post-trade profit ()",
                         "Total profit (cum) ()", "Units sold (#) (hidden)", "Units bought (#) (hidden)",
                        "Excess Final (#)", "Shortage Final (#)"] +
                        [f"{field}_TR{i}" for i in range(1, 6) for field in ("Type", "Amount", "Price")])
            if headers[:len(expected)] != expected:
                raise ValueError(f"unexpected headers in worksheet {sheet}")
            ignored = {"SimpleAvg_Price_TR", "WeightedAvg_Price_TR"}
            for name in headers[len(expected):]:
                if (name in ignored or re.fullmatch(r"(?:Type|Amount|Price)_TE\d+", name)
                        or re.fullmatch(r"Trade cash ?flow TE\d+", name)
                        or re.fullmatch(r"Trade cashflow TE\d+", name)):
                    continue
                raise ValueError(f"unexpected non-market-request column in worksheet {sheet}")
            index = {name: i for i, name in enumerate(headers)}
            treatment = "NV" if sheet.startswith("NV") else ("GA" if sheet.startswith("GA") else "G3")
            wholesale = 50 if sheet.endswith("50") else 10
            for row in rows[1:]:
                get = lambda field: row[index[field]] if index[field] < len(row) else ""
                player = str(get("Player")).strip()
                round_no = _number(get("Round"), "Round")
                if not player or round_no is None or not 1 <= round_no <= 15:
                    raise ValueError("invalid participant or formal round")
                key = (sheet, round_no, player)
                if key in seen:
                    raise ValueError("duplicate participant-round row")
                seen.add(key)
                p = _number(get("Sales Price"), "Sales Price")
                q = _number(get("Order"), "Order")
                if p is None or q is None or not 51 <= p <= 100 or not 0 <= q <= 400:
                    raise ValueError("paper procurement fields are outside documented ranges")
                episodes.append(Episode(player, sheet, treatment, round_no, "procurement",
                    {"p": p, "w": wholesale}, {"Q": q}))
                if treatment == "NV":
                    continue
                demand = _number(get("Actual Demand"), "Actual Demand")
                if demand is None or not 0 <= demand <= 200:
                    raise ValueError("paper demand outside documented range")
                orders = []
                for slot in range(1, 6):
                    side = str(get(f"Type_TR{slot}")).strip().upper()
                    amount = _number(get(f"Amount_TR{slot}"), f"Amount_TR{slot}", required=False)
                    price = _number(get(f"Price_TR{slot}"), f"Price_TR{slot}", required=False)
                    if not side and amount in (None, 0) and price in (None, 0):
                        continue
                    if side not in {"BUY", "SELL"} or amount is None or price is None:
                        raise ValueError("incomplete historical order triple")
                    orders.append({"type": side, "amount": amount, "price": price})
                episodes.append(Episode(player, sheet, treatment, round_no, "orders",
                    {"p": p, "w": wholesale, "Q": q, "D": demand}, {"orders": orders}))
        # A loaded workbook is never self-authorizing, even when checksum is valid.
        manifest = SourceManifest("human", "mnsc.2023.03771/Data_vf.xlsx", checksum,
            codebook_verified=codebook_verified, intended_use_authorized=intended_use_authorized,
            provider_transmission_permitted=provider_transmission_permitted)
        return manifest, episodes
    finally:
        archive.close()
