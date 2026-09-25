"""Generation 2 data: Binance USD-M daily positioning metrics (5-minute rows), checksum-verified.

Only pool symbols eligible at some decision in 2022-01..2026-08 are fetched. Each day's file is
reduced to the last row stamped strictly before 00:00 UTC (open-interest value, top-trader position ratio,
all-account ratio) and the day's mean log taker buy/sell ratio, stored as data/metrics_daily.parquet
(indexed by the decision day: the file of day d-1 serves the 00:00 decision of day d). The daily
archives are deleted after reduction; URLs and digests stay in data/download_ledger_metrics.json.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fasa.data import BASE, fetch_many, write_ledger
from fasa.panel import load
from fasa.signals import daily, eligible

RAW = ROOT / "data" / "raw_metrics"


def reduce_day(path: Path, before: pd.Timestamp) -> dict | None:
    """One daily archive -> its last row stamped strictly before ``before`` (00:00 UTC of the
    decision day, naive like ``create_time``) and the mean log taker ratio of the rows before it.
    Some files also hold a row stamped at or just after the next midnight (every SXPUSDT file from
    2024-03 until its settlement, and a few files in 2024-04); such a row is dropped."""
    with zipfile.ZipFile(path) as z:
        df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    if df.empty:
        return None
    df["create_time"] = pd.to_datetime(df["create_time"])
    df = df[df["create_time"] < before].sort_values("create_time")
    if df.empty:
        return None
    last = df.iloc[-1]
    tk = pd.to_numeric(df["sum_taker_long_short_vol_ratio"], errors="coerce")
    tk = tk[tk > 0]
    return {
        "oi_value": float(last["sum_open_interest_value"]),
        "top_pos_ratio": float(last["sum_toptrader_long_short_ratio"]),
        "all_ratio": float(last["count_long_short_ratio"]),
        "taker_log_mean": float(np.log(tk).mean()) if len(tk) else np.nan,
        "last_row_utc": str(last["create_time"]),
    }


def ledger_summary() -> None:
    """Public summary of the (large) metrics download ledger: per-symbol file counts and dates."""
    import hashlib
    import json

    path = ROOT / "data" / "download_ledger_metrics.json"
    led = json.loads(path.read_text())
    per = {}
    for r in led:
        s, d = r["url"].split("/")[-2], r["url"].split("-metrics-")[1][:10]
        e = per.setdefault(s, {"files": 0, "first": d, "last": d})
        e["files"] += 1
        e["first"], e["last"] = min(e["first"], d), max(e["last"], d)
    (ROOT / "data" / "metrics_ledger_summary.json").write_text(json.dumps({
        "source": "https://data.binance.vision/data/futures/um/daily/metrics/", "archives": len(led),
        "bytes": sum(r["bytes"] for r in led), "ledger_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "symbols": per}, indent=1))


def main() -> int:
    feat = daily(load(ROOT / "data" / "panel"))
    elig = eligible(feat)
    syms = sorted(elig.columns[elig.loc["2022-01-01":"2026-08-31"].any()])
    days = pd.date_range("2021-12-01", "2026-08-30", freq="D")
    jobs = [(f"{BASE}/futures/um/daily/metrics/{s}/{s}-metrics-{d:%Y-%m-%d}.zip",
             RAW / s / f"{s}-metrics-{d:%Y-%m-%d}.zip") for s in syms for d in days]
    print(f"{len(syms)} symbols x {len(days)} days = {len(jobs)} archives", flush=True)
    got = fetch_many(jobs, workers=48)
    write_ledger(ROOT / "data" / "download_ledger_metrics.json", got)
    ledger_summary()
    rows = []
    for s in syms:
        for f in sorted((RAW / s).glob("*.zip")):
            day = pd.Timestamp(f.stem.split("-metrics-")[1], tz="UTC") + pd.Timedelta(days=1)
            r = reduce_day(f, before=day.tz_localize(None))
            if r is None:
                continue
            rows.append({"date": day, "symbol": s, **r})
    out = pd.DataFrame(rows)
    out.to_parquet(ROOT / "data" / "metrics_daily.parquet")
    print(f"{len(got)} archives verified; {len(out)} symbol-days reduced", flush=True)
    for s in syms:
        for f in (RAW / s).glob("*.zip"):
            f.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
