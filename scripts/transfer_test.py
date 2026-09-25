"""G1b transfer test (docs/protocol_amendments.md): the frozen rule B_fund_L14_72h, once, on USD-M
USDT perpetuals OUTSIDE the study's 40-symbol pool, 2024-01-01..2026-08-31.

New universe: the 60 non-pool symbols with the largest 2023-12 quote volume (formed before the
test window); at each decision the top 15 by trailing 30-day volume are eligible, as in the study.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fasa import engine, evaluate, manifest, strategies
from fasa.data import fetch_many, list_um_symbols, months, read_klines, url, write_ledger
from fasa.panel import build
from fasa.signals import daily, eligible, residual

RAW = ROOT / "data" / "raw_transfer"
LO, HI = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-08-31", tz="UTC")


def main() -> int:
    tic = time.perf_counter()
    pool = set(json.loads((ROOT / "data" / "pool.json").read_text())["pool"])
    syms = [s for s in list_um_symbols(requests.Session()) if s.endswith("USDT") and "_" not in s and s not in pool]
    rows = fetch_many([(url("perp", s, "2023-12", "1d"), RAW / "formation" / f"{s}.zip") for s in syms])
    vol = {r["url"].split("/")[-3]: float(read_klines(RAW / "formation" / f"{r['url'].split('/')[-3]}.zip")["quote_volume"].sum())
           for r in rows}
    new = sorted(vol, key=vol.get, reverse=True)[:60]
    ms = months("2023-10", "2026-08")
    jobs = [(url("perp", s, m), RAW / "perp" / s / f"{s}-1h-{m}.zip") for s in new for m in ms]
    jobs += [(url("funding", s, m), RAW / "funding" / s / f"{s}-fundingRate-{m}.zip") for s in new for m in ms]
    got = fetch_many(jobs, workers=12)
    write_ledger(ROOT / "data" / "download_ledger_transfer.json", rows + got)
    (ROOT / "data" / "transfer_universe.json").write_text(json.dumps(
        {"rule": "non-pool USDT perps, top 60 by 2023-12 quote volume", "symbols": new,
         "quote_volume_2023_12": {s: vol[s] for s in new}}, indent=1))
    panel = build(RAW, new, "2023-10-01", "2026-08-31")
    feat = daily(panel)
    elig = eligible(feat)
    _, beta, mkt = residual(feat["ret"], elig, 1)
    tg = {k: v for k, v in strategies.funding_xs(panel, feat, elig, 14, "72h", beta).items() if LO <= k <= HI}
    slip = np.full(len(new), 5.0 / 1e4)
    led = engine.run(panel["perp"]["open"], panel["perp"]["close"], panel["funding"], tg,
                     engine.Costs(np.full(len(new), 5.0 / 1e4), slip), delay=1, band=0.02,
                     start=LO, end=HI + pd.Timedelta(hours=23))["ledger"]
    d = evaluate.daily_returns(led)
    dtb3 = pd.read_parquet(ROOT / "data" / "dtb3.parquet")["DTB3"]
    rf = evaluate.rf_daily(dtb3, feat["days"])
    m = evaluate.metrics(d, rf)
    m["mean_ci"] = evaluate.ci_mean(d["ret"])
    # market and momentum factors inside the new universe (zero cost)
    ret = feat["ret"]
    mk = ret.where(elig.shift(1)).mean(axis=1)
    mom = (1 + ret).rolling(21).apply(np.prod, raw=True) - 1
    mr = []
    for i, day in enumerate(feat["days"][:-1]):
        w = strategies.long_short(mom.loc[day].where(elig.loc[day]), beta.loc[day], w_cap=1.0, net_cap=1.0)
        mr.append((feat["days"][i + 1], float((w * ret.loc[feat["days"][i + 1]].fillna(0.0)).sum())))
    fac = pd.DataFrame({"MKT": mk, "MOM": pd.Series(dict(mr))}).loc[LO:HI]
    attr = evaluate.ols_hac(d["ret"], fac)
    quarters = d["ret"].groupby(d.index.to_period("Q")).sum()
    out = ROOT / "results" / "transfer"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "daily.csv")
    summ = {"rule": "B_fund_L14_72h", "universe_size": len(new), "metrics": m, "attribution_mkt_mom": attr,
            "quarters_positive": int((quarters > 0).sum()), "quarters": len(quarters),
            "eligible_symbols_ever": sorted(elig.columns[elig.loc[LO:HI].any()].tolist())}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    manifest.write(out / "manifest.json", {
        "study": "funding-aware-stat-arb / G1b transfer test", "command": "python scripts/transfer_test.py",
        "source_tree_sha256": manifest.tree_hash(ROOT),
        "protocol_sha256": manifest.sha256_file(ROOT / "configs" / "alpha_protocol.yaml"),
        "data_ledger_sha256": manifest.sha256_file(ROOT / "data" / "download_ledger_transfer.json"),
        "runtime_seconds": time.perf_counter() - tic}, [out / "daily.csv", out / "summary.json"])
    print(json.dumps({k: summ[k] for k in ("rule", "quarters_positive", "quarters")}, indent=1))
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})
    print({k: round(attr[k], 3) for k in ("alpha_ann", "alpha_t", "MKT_coef", "MOM_coef", "MOM_t", "r2")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
