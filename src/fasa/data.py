"""Binance public archives: listing, checksum-verified downloads and parsing.

Sources (data.binance.vision, no account or key): USD-M perpetual 1h klines and monthly funding
archives, spot 1h klines. Every archive is checked against its published ``.CHECKSUM`` (SHA-256)
before use and recorded in ``data/download_ledger.json``. Adapted from the checksum-verified
downloader of this portfolio's cross-market study (same conventions, rewritten standalone).
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.binance.vision/data"
LISTING = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
    "trades", "taker_base", "taker_quote", "ignore",
]  # fmt: skip


def url(kind: str, symbol: str, month: str, interval: str = "1h") -> str:
    """kind: 'perp' (USD-M klines), 'funding' (USD-M monthly funding) or 'spot' (klines)."""
    if kind == "perp":
        return f"{BASE}/futures/um/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    if kind == "spot":
        return f"{BASE}/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    if kind == "funding":
        return f"{BASE}/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"
    raise ValueError(kind)


def list_um_symbols(session: requests.Session) -> list[str]:
    """Every symbol with a USD-M monthly kline directory, delisted ones included."""
    out, marker = [], ""
    while True:
        r = session.get(
            LISTING,
            params={"delimiter": "/", "prefix": "data/futures/um/monthly/klines/", "marker": marker},
            timeout=60,
        )
        r.raise_for_status()
        found = re.findall(r"<Prefix>data/futures/um/monthly/klines/([^/<]+)/</Prefix>", r.text)
        out += found
        if "<IsTruncated>true</IsTruncated>" not in r.text or not found:
            return sorted(set(out))
        marker = f"data/futures/um/monthly/klines/{found[-1]}/"


def fetch(session: requests.Session, u: str, dest: Path, tries: int = 4) -> dict | None:
    """Download ``u`` to ``dest`` if missing and verify its SHA-256; None if the file does not
    exist on the server (404). Finite retries for transient failures."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        try:
            if not dest.exists():
                r = session.get(u, timeout=60)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                dest.write_bytes(r.content)
            c = session.get(u + ".CHECKSUM", timeout=60)
            c.raise_for_status()
            expected = c.text.split()[0].strip().lower()
            got = hashlib.sha256(dest.read_bytes()).hexdigest()
            if got != expected:
                dest.unlink()
                raise OSError(f"checksum mismatch for {u}")
            return {
                "url": u,
                "sha256": got,
                "bytes": dest.stat().st_size,
                "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        except (requests.RequestException, OSError):
            if attempt == tries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    return None


def fetch_many(jobs: list[tuple[str, Path]], workers: int = 8) -> list[dict]:
    """Parallel checksum-verified downloads; returns ledger rows of the files that exist."""
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(lambda j: fetch(session, *j), jobs))
    return [r for r in rows if r is not None]


def read_klines(path: Path) -> pd.DataFrame:
    """One monthly kline archive -> frame indexed by open time (UTC). Timestamps in the archives
    switched from milliseconds to microseconds in 2025 for spot; both are handled."""
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    first = raw.split(b"\n", 1)[0]
    header = 0 if first[:4].isalpha() or b"open_time" in first else None
    df = pd.read_csv(io.BytesIO(raw), header=header, names=None if header == 0 else KLINE_COLS)
    df.columns = KLINE_COLS[: len(df.columns)]
    ot = pd.to_numeric(df["open_time"])
    unit = "us" if ot.iloc[0] > 1e14 else "ms"
    df.index = pd.to_datetime(ot, unit=unit, utc=True)
    return df[["open", "close", "quote_volume"]].astype(float)


def read_funding(path: Path) -> pd.DataFrame:
    """Monthly funding archive -> settlements (time, interval hours, rate)."""
    with zipfile.ZipFile(path) as z:
        df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    t = pd.to_datetime(pd.to_numeric(df["calc_time"]), unit="ms", utc=True)
    return pd.DataFrame(
        {
            "time": t,
            "interval_h": pd.to_numeric(df["funding_interval_hours"]).astype(int),
            "rate": pd.to_numeric(df["last_funding_rate"]).astype(float),
        }
    )


def months(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def write_ledger(path: Path, rows: list[dict]) -> None:
    old = json.loads(path.read_text()) if path.exists() else []
    seen = {r["url"] for r in rows}
    path.write_text(json.dumps([r for r in old if r["url"] not in seen] + rows, indent=0))
