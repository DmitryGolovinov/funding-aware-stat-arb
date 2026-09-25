"""Download the study's public data (checksum-verified) and form the fixed symbol pool.

1. List every USD-M symbol (delisted included) and keep USDT-quoted perpetuals.
2. Pool formation (configs/alpha_protocol.yaml): the 40 with the largest 2020-12 quote volume
   among those with a 2020-12 archive; later listings are excluded.
3. Pool 1h perp klines and monthly funding from the warm-up month to the final end; BTCUSDT and
   ETHUSDT 1h spot klines for the carry family.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fasa.data import fetch_many, list_um_symbols, months, read_klines, url, write_ledger

RAW = ROOT / "data" / "raw"


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "alpha_protocol.yaml").read_text())
    session = requests.Session()
    syms = [s for s in list_um_symbols(session) if s.endswith("USDT") and "_" not in s]
    print(f"{len(syms)} USDT-quoted USD-M perpetual symbols listed (delisted included)")
    jobs = [(url("perp", s, "2020-12", "1d"), RAW / "pool" / f"{s}-1d-2020-12.zip") for s in syms]
    rows = fetch_many(jobs)
    vol = {}
    for r in rows:
        s = r["url"].split("/")[-3]
        vol[s] = float(read_klines(RAW / "pool" / f"{s}-1d-2020-12.zip")["quote_volume"].sum())
    ranked = sorted(vol, key=vol.get, reverse=True)
    pool = ranked[:40]
    (ROOT / "data" / "pool.json").write_text(
        json.dumps(
            {
                "rule": cfg["pool"]["formation"],
                "candidates_with_2020_12_archive": len(vol),
                "pool": pool,
                "quote_volume_2020_12": {s: vol[s] for s in pool},
            },
            indent=1,
        )
    )
    print("pool:", " ".join(pool))
    ms = months(cfg["calendar"]["warmup_from"], cfg["calendar"]["final"][1])
    jobs = [(url("perp", s, m), RAW / "perp" / s / f"{s}-1h-{m}.zip") for s in pool for m in ms]
    jobs += [(url("funding", s, m), RAW / "funding" / s / f"{s}-fundingRate-{m}.zip") for s in pool for m in ms]
    jobs += [(url("spot", s, m), RAW / "spot" / s / f"{s}-1h-{m}.zip") for s in ("BTCUSDT", "ETHUSDT") for m in ms]
    got = fetch_many(jobs, workers=12)
    fred = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3&cosd=2020-01-01&coed=2026-09-01"
    rf = pd.read_csv(io.StringIO(session.get(fred, timeout=60).text))
    rf = rf.set_index(rf.columns[0])[rf.columns[1]].apply(pd.to_numeric, errors="coerce")
    rf.index = pd.to_datetime(rf.index)
    rf.to_frame("DTB3").to_parquet(ROOT / "data" / "dtb3.parquet")
    write_ledger(ROOT / "data" / "download_ledger.json", rows + got)
    total = sum(r["bytes"] for r in rows + got)
    print(f"{len(got)} archives of {len(jobs)} requested exist; {total / 1e6:.0f} MB verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
