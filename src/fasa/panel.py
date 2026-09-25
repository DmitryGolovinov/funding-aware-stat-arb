"""Hourly panel: perp and spot open/close/quote volume on one UTC hour grid, plus settled funding.

Prices are the traded 1h klines (no invented index). Missing hours stay NaN (never filled with
zeros); a symbol with no bars after some date is treated as delisted from that date.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import read_funding, read_klines


def build(raw: Path, pool: list[str], start: str, end: str) -> dict:
    grid = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h", tz="UTC")
    out = {}
    for kind, syms in (("perp", pool), ("spot", ["BTCUSDT", "ETHUSDT"])):
        fields = {f: pd.DataFrame(np.nan, index=grid, columns=syms) for f in ("open", "close", "qv")}
        for s in syms:
            files = sorted((raw / kind / s).glob("*.zip"))
            if not files:
                continue
            k = pd.concat([read_klines(f) for f in files]).sort_index()
            k = k[~k.index.duplicated(keep="first")].reindex(grid)
            fields["open"][s], fields["close"][s], fields["qv"][s] = k["open"], k["close"], k["quote_volume"]
        out[kind] = fields
    ev = []
    for s in pool:
        for f in sorted((raw / "funding" / s).glob("*.zip")):
            d = read_funding(f)
            d["symbol"] = s
            ev.append(d)
    fund = pd.concat(ev, ignore_index=True).drop_duplicates(["symbol", "time"]).sort_values("time")
    fund = fund[(fund["time"] >= grid[0]) & (fund["time"] <= grid[-1])]
    out["funding"] = fund.reset_index(drop=True)
    return settle_tails(out)


def settle_tails(panel: dict, min_gap_hours: int = 48) -> dict:
    """Cut the dead tail of settled contracts.

    Binance keeps publishing hourly klines for some perpetuals after they were settled and
    delisted (for example SXPUSDT: automatic settlement 2025-12-05 09:00 UTC per the exchange's
    announcement; the archive continues with a frozen price and almost no volume). A live
    perpetual pays funding on every settlement, so when bars continue more than ``min_gap_hours``
    after a symbol's last funding settlement, every bar from one hour after that settlement on is
    set to missing. The ledger then closes a held position at the last close before the cut, and
    the symbol stops being eligible. Returns the list of cuts in ``panel["settlements"]``."""
    f = panel["funding"]
    last_f = f.groupby("symbol")["time"].max()
    cuts = {}
    for s in panel["perp"]["open"].columns:
        o = panel["perp"]["open"][s]
        lb = o.last_valid_index()
        if s not in last_f.index or lb is None:
            continue
        end = last_f[s].floor("h") + pd.Timedelta(hours=1)
        if lb - last_f[s] > pd.Timedelta(hours=min_gap_hours):
            for fld in ("open", "close", "qv"):
                panel["perp"][fld].loc[end:, s] = np.nan
            cuts[s] = str(end)
    panel["settlements"] = cuts
    return panel


def save(panel: dict, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for kind in ("perp", "spot"):
        for f, df in panel[kind].items():
            df.to_parquet(dest / f"{kind}_{f}.parquet")
    panel["funding"].to_parquet(dest / "funding.parquet")


def load(src: Path) -> dict:
    out = {k: {f: pd.read_parquet(src / f"{k}_{f}.parquet") for f in ("open", "close", "qv")} for k in ("perp", "spot")}
    out["funding"] = pd.read_parquet(src / "funding.parquet")
    return settle_tails(out)
